import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.normal import Normal

from model.gumbel import gumbel_sigmoid
from model.modules import ChannelMHAttention, ChannelMaskValue
from model.inference_utils import reset_layer, forward_network
from model.dynamics_utils import get_train_mask, get_eval_mask

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class DynamicsGradAttn(nn.Module):
    def __init__(self, encoder, obj_encoder, params):
        super(DynamicsGradAttn, self).__init__()

        self.encoder = encoder
        self.obj_encoder = obj_encoder

        self.params = params
        self.device = device = params.device
        self.dynamics_params = dynamics_params = params.dynamics_params

        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action
        self.parallel_sample = params.training_params.replay_buffer_params.prioritized_buffer

        self.dynamics_params = dynamics_params = params.dynamics_params
        self.num_pred_steps = dynamics_params.num_pred_steps

        self.contrastive_params = contrastive_params = dynamics_params.contrastive_params
        self.num_negative_samples = contrastive_params.num_negative_samples
        self.num_pred_samples = contrastive_params.num_pred_samples
        self.num_pred_iters = contrastive_params.num_pred_iters
        self.pred_sigma_init = contrastive_params.pred_sigma_init
        self.pred_sigma_shrink = contrastive_params.pred_sigma_shrink

        # (feature_dim,)
        if self.continuous_state:
            self.delta_feature_min = self.encoder({key: val[0] for key, val in self.params.obs_delta_range.items()})
            self.delta_feature_max = self.encoder({key: val[1] for key, val in self.params.obs_delta_range.items()})

        self.grad_attn_params = params.dynamics_params.grad_attn_params
        self.energy_norm_reg_coef = self.grad_attn_params.energy_norm_reg_coef
        self.delta_grad_reg_coef = self.grad_attn_params.delta_grad_reg_coef

        self.init_model()
        self.reset_params()
        self.init_graph()
        self.reset_mask_eval()

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=dynamics_params.lr)

        self.load(params.training_params.load_dynamics, device)
        self.train()

        self.tau = 0.01
        self.sa_grad_lt_thre_reg_coef = 0
        self.epsilon = 1e-6
        self.feature = None

    def init_model(self):
        params = self.params
        grad_attn_params = self.grad_attn_params
        obj_encoder = self.obj_encoder

        # model params
        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = params.feature_dim
        self.num_objs = num_objs = obj_encoder.num_objs
        self.obj_dim = obj_dim = obj_encoder.obj_dim
        self.input_dim = input_dim = max(action_dim, max(self.obj_dim.values()))
        self.query_index = obj_encoder.obs2obj_index

        self.sa_feature_weights = nn.ParameterList()
        self.sa_feature_biases = nn.ParameterList()

        # for instance-wise mask
        self.sa_mask_weights = nn.ParameterList()
        self.sa_mask_biases = nn.ParameterList()

        # for implicit (contrative) dynamics
        self.sa_encoder_weights = nn.ParameterList()
        self.sa_encoder_biases = nn.ParameterList()
        self.d_encoder_weights = nn.ParameterList()
        self.d_encoder_biases = nn.ParameterList()

        # for explicit dynamics
        self.predictor_weights = nn.ParameterList()
        self.predictor_biases = nn.ParameterList()

        # Instantiate the parameters of each module in each variable's dynamics model
        # state action feature extractor
        in_dim = input_dim
        for out_dim in grad_attn_params.feature_fc_dims:
            self.sa_feature_weights.append(nn.Parameter(torch.zeros(feature_dim * (num_objs + 1), in_dim, out_dim)))
            self.sa_feature_biases.append(nn.Parameter(torch.zeros(feature_dim * (num_objs + 1), 1, out_dim)))
            in_dim = out_dim

        feature_embed_dim = grad_attn_params.feature_fc_dims[-1]

        attn_params = grad_attn_params.attn_params
        attn_dim = attn_params.attn_dim
        num_heads = attn_params.num_heads
        attn_out_dim = attn_params.attn_out_dim
        use_bias = attn_params.attn_use_bias
        residual = attn_params.residual
        share_weight_across_kqv = attn_params.share_weight_across_kqv
        post_fc_dims = attn_params.post_fc_dims
        if post_fc_dims:
            assert attn_out_dim == post_fc_dims[-1]

        self.attns = nn.ModuleList()
        in_dim = feature_embed_dim
        for i in range(grad_attn_params.num_attns):
            num_queries = num_keys = num_objs + 1
            in_dim = attn_out_dim if i else feature_embed_dim
            attn = ChannelMHAttention(feature_dim, attn_dim, num_heads, num_queries, in_dim, num_keys, in_dim,
                                      out_dim=attn_out_dim, use_bias=use_bias, residual=residual,
                                      share_weight_across_kqv=share_weight_across_kqv, post_fc_dims=post_fc_dims)
            self.attns.append(attn)

        self.mask_params = mask_params = grad_attn_params.mask_params
        self.mask_type = mask_params.mask_type
        assert self.mask_type in ["none", "sample", "input", "attention"]

        if self.mask_type in ["input", "attention"]:
            in_dim = feature_embed_dim
            num_queries = num_keys = num_objs + 1

            self.learn_signature_queries = mask_params.learn_signature_queries
            if self.learn_signature_queries:
                b = 1 / np.sqrt(in_dim)
                self.signature_queries = nn.Parameter(torch.FloatTensor(feature_dim, num_queries, in_dim).uniform_(-b, b))

            self.learn_bandwidth = mask_params.learn_bandwidth
            if mask_params.learn_bandwidth:
                self.bandwidth = nn.Parameter(torch.tensor(np.log(mask_params.init_bandwidth), dtype=torch.float32))
            else:
                self.bandwidth = mask_params.init_bandwidth

            if self.mask_type == "input":
                mask_final_dim = 1
            elif self.mask_type == "attention":
                mask_final_dim = mask_params.signature_dim
            else:
                raise NotImplementedError

            self.mask_attns = nn.ModuleList()
            for i in range(mask_params.num_mask_attns):
                in_dim = attn_out_dim if i else in_dim
                attn = ChannelMHAttention(feature_dim, attn_dim, num_heads, num_queries, in_dim, num_keys, in_dim,
                                          out_dim=attn_out_dim, use_bias=use_bias, residual=residual,
                                          share_weight_across_kqv=share_weight_across_kqv, post_fc_dims=post_fc_dims)
                self.mask_attns.append(attn)

            in_dim = attn_out_dim
            for out_dim in mask_params.mask_fc_dims + [mask_final_dim]:
                self.sa_mask_weights.append(nn.Parameter(torch.zeros(feature_dim * (num_objs + 1), in_dim, out_dim)))
                self.sa_mask_biases.append(nn.Parameter(torch.zeros(feature_dim * (num_objs + 1), 1, out_dim)))
                in_dim = out_dim

        self.sampled_sa_mask = None

        self.dynamics_mode = grad_attn_params.dynamics_mode
        assert self.dynamics_mode in ["contrastive", "pred_scalar", "pred_normal"]
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # sa_feature encoder
            in_dim = attn_out_dim
            for out_dim in grad_attn_params.sa_encoding_fc_dims:
                self.sa_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.sa_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

            # delta feature encoder
            in_dim = 1
            for out_dim in grad_attn_params.delta_encoding_fc_dims:
                self.d_encoder_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.d_encoder_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim
        else:
            # predictor
            in_dim = attn_out_dim
            for out_dim in grad_attn_params.predictor_fc_dims:
                self.predictor_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.predictor_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
                in_dim = out_dim

            if not self.continuous_state:
                self.feature_inner_dim = obj_encoder.feature_inner_dim
                final_size = max(self.feature_inner_dim)

                self.logits_mask = torch.zeros(feature_dim, final_size, dtype=torch.bool, device=self.device)
                for i, feature_i_inner_dim in enumerate(self.feature_inner_dim):
                    self.logits_mask[i, :feature_i_inner_dim] = True

            elif self.dynamics_mode == "pred_scalar":
                final_size = 1
            else:
                final_size = 2

            self.predictor_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, final_size)))
            self.predictor_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, final_size)))

    def reset_params(self):
        feature_dim = self.feature_dim
        module_weights = [self.sa_feature_weights,
                          self.sa_mask_weights,
                          self.sa_encoder_weights,
                          self.d_encoder_weights,
                          self.predictor_weights]
        module_biases = [self.sa_feature_biases,
                         self.sa_mask_biases,
                         self.sa_encoder_biases,
                         self.d_encoder_biases,
                         self.predictor_biases]
        for weights, biases in zip(module_weights, module_biases):
            for w, b in zip(weights, biases):
                assert w.ndim == b.ndim == 3
                for i in range(w.shape[0]):
                    reset_layer(w[i], b[i])

    def init_graph(self):
        feature_dim = self.feature_dim
        action_dim = self.action_dim if self.continuous_action else 1
        device = self.device
        self.global_causality_threshold = self.grad_attn_params.global_causality_threshold

        if self.mask_type == "sample":
            num_cols = self.num_objs + 1
        else:
            num_cols = feature_dim + action_dim
        self.global_causality_val = torch.ones(feature_dim, num_cols, device=device) * self.global_causality_threshold
        self.global_causality_mask = torch.ones_like(self.global_causality_val, dtype=torch.bool)

    @staticmethod
    def compute_annealing_val(step, starts=0, start_val=0, ends=0, end_val=0):
        coef = np.clip((step - starts) / (ends - starts), 0, 1)
        return start_val + coef * (end_val - start_val)

    def setup_annealing(self, step):
        starts = self.grad_attn_params.sa_grad_reg_anneal_starts
        ends = self.grad_attn_params.sa_grad_reg_anneal_ends
        end_val = self.grad_attn_params.sa_grad_lt_thre_reg_coef
        self.sa_grad_lt_thre_reg_coef = self.compute_annealing_val(step, starts, 0, ends, end_val)
        end_val = self.grad_attn_params.sa_grad_ge_thre_reg_coef
        self.sa_grad_ge_thre_reg_coef = self.compute_annealing_val(step, starts, 0, ends, end_val)

        starts = self.mask_params.tau_anneal_starts
        start_val = self.mask_params.tau_start_val
        ends = self.mask_params.tau_anneal_ends
        end_val = self.mask_params.tau_end_val
        self.tau = self.compute_annealing_val(step, starts, start_val, ends, end_val)

        starts = self.mask_params.mask_reg_anneal_starts
        ends = self.mask_params.mask_reg_anneal_ends
        end_val = self.mask_params.mask_reg_coef
        self.mask_reg_coef = self.compute_annealing_val(step, starts, 0, ends, end_val)

    def sample_delta_feature(self, shape, num_samples):
        # (bs, num_pred_samples, feature_dim)
        uniform_noise = torch.rand(*shape, num_samples, self.feature_dim, dtype=torch.float32, device=self.device)
        delta_feature = uniform_noise * (self.delta_feature_max - self.delta_feature_min) + self.delta_feature_min
        return delta_feature

    def extract_state_action_feature(self, obj_feature, action):
        """
        :param obj_feature:
            if self.parallel_sample and self.training: [(feature_dim, bs, obj_i_dim)] * num_objs.
            else: [(bs, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
        :return: (feature_dim, num_objs + 1, bs, out_dim),
        """
        feature_dim = self.feature_dim
        num_objs = self.num_objs

        if not (self.parallel_sample and self.training):
            # [(feature_dim, bs, obj_i_dim)] * num_objs
            obj_feature = [obj_feature_i.unsqueeze(dim=0).repeat(feature_dim, 1, 1)
                           for obj_feature_i in obj_feature]
            action = action.unsqueeze(dim=0)                        # (1, bs, action_dim)
            action = action.repeat(feature_dim, 1, 1)               # (feature_dim, bs, action_dim)

        bs = action.shape[1]

        # inputs: [(feature_dim, bs, input_i_dim)] * (num_objs + 1)
        inputs, self.sa_inputs = [], []
        for input_i in obj_feature + [action]:
            input_i = input_i.detach()
            input_i.requires_grad = True
            self.sa_inputs.append(input_i)

            input_i = input_i.permute(2, 0, 1)                      # (input_i_dim, feature_dim, bs)
            inputs.append(input_i)

        x = pad_sequence(inputs)                                    # (input_dim, num_objs + 1, feature_dim, bs)
        x = x.permute(2, 1, 3, 0)                                   # (feature_dim, num_objs + 1, bs, input_dim)
        x = x.reshape(feature_dim * (num_objs + 1), bs, -1)         # (feature_dim * (num_objs + 1), bs, input_dim)

        sa_feature = forward_network(x, self.sa_feature_weights, self.sa_feature_biases)
        sa_feature = F.relu(sa_feature)
        sa_feature = sa_feature.reshape(feature_dim, num_objs + 1, bs, -1)

        return sa_feature                                           # (feature_dim, num_objs + 1, bs, out_dim)

    def extract_sa_mask(self, sa_feature):
        log_attn_mask = None
        self.input_mask = None
        self.expand_bs = False
        feature_dim, num_objs_p_1, bs, _ = sa_feature.shape
        if self.mask_type in ["input", "attention"]:
            sa_mask_feature = sa_feature

            if not self.training or self.sa_grad_lt_thre_reg_coef > 0:
                # detach so that partial gradient regularization doesn't penalize the mask computation
                sa_mask_feature = sa_mask_feature.detach()

            if self.learn_signature_queries:
                # (feature_dim, num_objs + 1, out_dim) -> (feature_dim, num_objs + 1, bs, out_dim)
                queries = self.signature_queries.unsqueeze(dim=2).expand(-1, -1, bs, -1)
            else:
                queries = sa_mask_feature

            for attn in self.mask_attns:
                # (feature_dim, num_objs + 1, bs, attn_out_dim)
                sa_mask_feature = attn(queries, sa_mask_feature)
                queries = sa_mask_feature

            # (feature_dim * (num_objs + 1), bs, attn_out_dim)
            sa_mask_feature = sa_mask_feature.view(feature_dim * num_objs_p_1, bs, -1)
            # (feature_dim * (num_objs + 1), bs, 1)
            sa_mask_logit = forward_network(sa_mask_feature, self.sa_mask_weights, self.sa_mask_biases)
            sa_mask_logit = sa_mask_logit.view(feature_dim, num_objs_p_1, bs, -1)

            self.sa_feature = sa_feature
            self.queries = queries
            self.sa_mask_feature = sa_mask_feature
            self.sa_mask_logit = sa_mask_logit

            if self.mask_type == "input":
                if self.training:
                    self.input_mask = gumbel_sigmoid(sa_mask_logit, tau=self.tau)
                else:
                    self.input_mask = (sa_mask_logit > 0).float()
                sa_feature = sa_feature * self.input_mask
            else:
                # (feature_dim, num_objs + 1, bs, signature_dim)
                attn_signatures = F.normalize(sa_mask_logit, p=2, dim=-1)
                attn_signatures_q = attn_signatures.permute(0, 2, 1, 3).reshape(feature_dim * bs, num_objs_p_1, -1)
                attn_signatures_k = attn_signatures.permute(0, 2, 3, 1).reshape(feature_dim * bs, -1, num_objs_p_1)
                # (feature_dim * bs, num_objs + 1, num_objs + 1)
                dist = 1 - torch.bmm(attn_signatures_q, attn_signatures_k)

                dist = dist.view(feature_dim, bs, num_objs_p_1, num_objs_p_1)
                if self.learn_bandwidth:
                    bandwidth = torch.exp(self.bandwidth).clamp(min=1e-6)
                else:
                    bandwidth = self.bandwidth
                log_P_qk = -dist / bandwidth
                if self.training:
                    K_qk = gumbel_sigmoid(log_P_qk, tau=self.tau, hard=False)
                else:
                    K_qk = torch.exp(log_P_qk)
                log_attn_mask = K_qk / (K_qk.sum(dim=-1, keepdim=True) + self.epsilon)
                log_attn_mask = torch.log(log_attn_mask)

                self.log_P_qk = log_P_qk
                self.K_qk = K_qk
                self.log_attn_mask = log_attn_mask

        elif self.mask_type == "sample":
            assert self.sampled_sa_mask is not None
            if self.sampled_sa_mask.ndim == 3:
                # self.sampled_sa_mask: (feature_dim, bs, num_objs + 1)
                # sampled_sa_mask: (feature_dim, num_objs + 1, bs, 1)
                sampled_sa_mask = self.sampled_sa_mask.permute(0, 2, 1).unsqueeze(dim=-1)
                sa_feature = sa_feature * sampled_sa_mask
            elif self.sampled_sa_mask.ndim == 4:
                self.expand_bs = True
                # self.sampled_sa_mask: (feature_dim, bs, num_objs + 2, num_objs + 1)
                # sampled_sa_mask: (feature_dim, num_objs + 1, bs, num_objs + 2, 1)
                sampled_sa_mask = self.sampled_sa_mask.permute(0, 3, 1, 2).unsqueeze(dim=-1)
                # (feature_dim, num_objs + 1, bs, 1, out_dim)
                sa_feature = sa_feature.unsqueeze(dim=-2)
                sa_feature = sa_feature * sampled_sa_mask
                sa_feature = sa_feature.view(feature_dim, num_objs_p_1, bs * (num_objs_p_1 + 1), -1)
            else:
                raise NotImplementedError

        return sa_feature, log_attn_mask

    def extract_sa_encoding(self, obj_feature, action):
        """
        :param obj_feature:
            if self.parallel_sample and self.training: [(feature_dim, bs, obj_i_dim)] * num_objs.
            else: [(bs, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
        :return: (feature_dim, bs, encoding_dim)
        """
        feature_dim = self.feature_dim
        num_objs = self.num_objs

        # (feature_dim, num_objs + 1, bs, out_dim)
        sa_feature = self.extract_state_action_feature(obj_feature, action)
        bs = sa_feature.shape[2]

        # (feature_dim, num_objs + 1, bs, out_dim), (feature_dim, bs, num_objs + 1, num_objs + 1)
        sa_feature, log_attn_mask = self.extract_sa_mask(sa_feature)

        if not self.evaluate_attn_mask:
            for attn in self.attns:
                # (feature_dim, num_objs + 1, bs, attn_out_dim)
                sa_feature = attn(sa_feature, sa_feature, log_attn_mask=log_attn_mask)
        else:
            attn_mask = None
            attn_signature_mask = None
            for attn in self.attns:
                # (feature_dim, num_objs + 1, bs, attn_out_dim), (bs, feature_dim, num_heads, num_queries, num_keys)
                sa_feature, attn_score = attn(sa_feature, sa_feature, return_attn=True, log_attn_mask=log_attn_mask)
                attn_score = attn_score.mean(dim=2).view(bs * feature_dim, num_objs + 1, num_objs + 1)
                if attn_mask is None:
                    attn_mask = attn_score
                else:
                    attn_mask = torch.bmm(attn_score, attn_mask)

                if self.mask_type == "attention":
                    if attn_signature_mask is None:
                        attn_signature_mask = torch.exp(log_attn_mask)
                        attn_signature_mask = attn_signature_mask.view(feature_dim * bs, num_objs + 1, num_objs + 1)
                    else:
                        attn_signature_mask = torch.bmm(attn_signature_mask, attn_signature_mask)

            attn_mask = attn_mask.view(bs, feature_dim, num_objs + 1, num_objs + 1)
            self.attn_mask = attn_mask[:, np.arange(self.feature_dim), self.query_index]
            if self.mask_type == "attention":
                attn_signature_mask = attn_signature_mask.view(feature_dim, bs, num_objs + 1, num_objs + 1)
                attn_signature_mask = attn_signature_mask.permute(1, 0, 2, 3)
                self.attn_signature_mask = attn_signature_mask[:, np.arange(self.feature_dim), self.query_index]

        # (feature_dim, bs, out_dim)
        sa_feature = sa_feature[np.arange(self.feature_dim), self.query_index]

        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (feature_dim, bs, encoding_dim)
            sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)
        else:
            # (feature_dim, bs, 1 / 2)
            sa_encoding = forward_network(sa_feature, self.predictor_weights, self.predictor_biases)

        if self.expand_bs:
            sa_encoding = sa_encoding.view(feature_dim, bs, num_objs + 2, -1)

        self.sa_feature_after_attn = sa_feature
        self.sa_encoding = sa_encoding

        return sa_encoding

    def extract_delta_state_encoding(self, delta_feature):
        """
        :param delta_feature: (bs, num_samples, feature_dim)
            notice that bs must be 1D
        :return: (feature_dim, bs, num_samples, encoding_dim)
        """
        feature_dim = self.feature_dim
        bs, num_samples, _ = delta_feature.shape
        x = delta_feature.view(-1, feature_dim).T                   # (feature_dim, bs * num_samples)
        x = x.unsqueeze(dim=-1)                                     # (feature_dim, bs * num_samples, 1)

        # (feature_dim, bs * num_samples, encoding_dim)
        delta_encoding = forward_network(x, self.d_encoder_weights, self.d_encoder_biases)
        # (feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = delta_encoding.view(feature_dim, bs, num_samples, -1)

        return delta_encoding                                       # (feature_dim, bs, num_samples, encoding_dim)

    def forward_step(self, obj_feature, action, delta_features=None):
        """
        :param obj_feature:
            if self.parallel_sample and self.training: [(feature_dim, bs, obj_i_dim)] * num_objs.
            else: [(bs, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
        :param delta_features: (bs, num_samples, feature_dim)
        :return: energy: (bs, num_samples, feature_dim)
        """
        # extract encodings from the action & state variables
        sa_encoding = self.extract_sa_encoding(obj_feature, action)

        if self.dynamics_mode != "contrastive":
            if sa_encoding.ndim == 3:
                # if self.dynamics_mode == "pred_scalar": (bs, feature_dim, 1)
                # elif self.dynamics_mode == "pred_normal": (bs, feature_dim, 2)
                prediction = sa_encoding.permute(1, 0, 2)
            elif sa_encoding.ndim == 4:
                # if self.dynamics_mode == "pred_scalar": (bs, num_objs + 2, feature_dim, 1)
                # elif self.dynamics_mode == "pred_normal": (bs, num_objs + 2, feature_dim, 2)
                prediction = sa_encoding.permute(1, 2, 0, 3)
            else:
                raise NotImplementedError
            return prediction

        delta_features = delta_features.detach()
        delta_features.requires_grad = True
        self.input_delta_features = delta_features

        # (feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = self.extract_delta_state_encoding(delta_features)

        if self.expand_bs:
            sa_encoding = sa_encoding.unsqueeze(dim=-2)             # (feature_dim, bs, num_objs + 2, 1, encoding_dim)
            delta_encoding = delta_encoding.unsqueeze(dim=-3)       # (feature_dim, bs, 1, num_samples, encoding_dim)
            energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (feature_dim, bs, num_objs + 2, num_samples)
            energy = energy.permute(1, 2, 3, 0)                     # (bs, num_objs + 2, num_samples, feature_dim)
        else:
            sa_encoding = sa_encoding.unsqueeze(dim=-2)             # (feature_dim, bs, 1, encoding_dim)
            energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (feature_dim, bs, num_samples)
            energy = energy.permute(1, 2, 0)                        # (bs, num_samples, feature_dim)

        return energy

    def forward_with_feature(self, obj_feature, feature, action, next_feature=None, neg_delta_feature=None):
        """
        :param obj_feature:
            if self.parallel_sample and self.training: [(feature_dim, bs, obj_i_dim)] * num_objs.
            else: [(bs, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param feature:
            if self.parallel_sample and self.training: (feature_dim, bs, feature_dim)
            else: (bs, feature_dim)
        :param action:
            if self.parallel_sample and self.training: (feature_dim, bs, action_dim)
            else: (bs, action_dim)
        :param next_feature:
            if self.parallel_sample and self.training: (feature_dim, bs, feature_dim)
            else: (bs, feature_dim)
        :param neg_delta_feature: (bs, num_negative_samples, feature_dim)
        :return: energy: (bs, 1 + num_negative_samples, feature_dim)
        """
        feature_dim = self.feature_dim
        action_dim = self.action_dim

        if not self.continuous_action:
            action = F.one_hot(action.squeeze(dim=-1), action_dim).float()      # (bs, action_dim)

        bs = action.shape[:-1]
        if self.parallel_sample and self.training:
            bs = bs[1:]

        flatten_bs = len(bs) > 1
        if flatten_bs:
            if self.sampled_sa_mask is not None or (self.parallel_sample and self.training):
                raise NotImplementedError

            obj_feature = [obj_feature_i.view(-1, obj_feature_i.shape[-1]) for obj_feature_i in obj_feature]
            action = action.reshape(-1, action_dim)

            if self.continuous_state:
                feature = feature.view(-1, feature_dim)
                if next_feature is not None:
                    next_feature = next_feature.view(-1, feature_dim)

                if neg_delta_feature is not None:
                    num_negative_samples = neg_delta_feature.shape[-2]
                    neg_delta_feature = neg_delta_feature.view(-1, num_negative_samples, feature_dim)
            else:
                feature = [feature_i.view(-1, feature_i.shape[-1]) for feature_i in feature]

        if not self.continuous_state:
            logits = self.forward_step(obj_feature, action)                             # (bs, feature_dim, final_size)
            logits[..., ~self.logits_mask] = -np.inf

            if flatten_bs:
                logits = logits.view(*bs, *logits.shape[-2:])

            return logits
        elif self.dynamics_mode == "contrastive":
            assert neg_delta_feature is not None

            if next_feature is not None:
                delta_feature = (next_feature - feature).detach()                       # (bs, feature_dim)

                if self.parallel_sample and self.training:
                    eye = torch.eye(feature_dim, device=self.device).unsqueeze(dim=-2)
                    delta_feature = (delta_feature * eye).sum(dim=-1).T                 # (bs, feature_dim)

                delta_feature = delta_feature.unsqueeze(dim=-2)                         # (bs, 1, feature_dim)
                # (bs, 1 + num_negative_samples, feature_dim)
                delta_features = torch.cat([delta_feature, neg_delta_feature], dim=-2)
            else:
                delta_features = neg_delta_feature

            # (bs, 1 + num_negative_samples, feature_dim)
            energy = self.forward_step(obj_feature, action, delta_features)

            if flatten_bs:
                energy = energy.view(*bs, *energy.shape[-2:])

            return energy
        else:
            # if self.dynamics_mode == "pred_scalar": (bs, feature_dim, 1)
            # if self.dynamics_mode == "pred_normal": (bs, feature_dim, 2)
            prediction = self.forward_step(obj_feature, action)

            if self.parallel_sample and self.training:
                eye = torch.eye(feature_dim, device=self.device).unsqueeze(dim=-2)
                feature = (feature * eye).sum(dim=-1).T                                 # (bs, feature_dim)

            if self.expand_bs:
                feature = feature.unsqueeze(dim=1).expand(-1, self.num_objs + 2, -1)

            if self.evaluate_mixture_cmi_mask and self.feature is not None:
                self.feature = self.feature.view(*feature.shape)
                feature = self.feature

            if self.dynamics_mode == "pred_scalar":
                prediction = prediction[..., 0]                                         # (bs, feature_dim)
                prediction = prediction + feature

                if flatten_bs:
                    prediction = prediction.view(*bs, feature_dim)
            else:
                mu, log_std = prediction.unbind(dim=-1)
                mu = mu + feature
                std = F.softplus(log_std).clamp(min=1e-6)

                if flatten_bs:
                    mu = mu.view(*bs, feature_dim)
                    std = std.view(*bs, feature_dim)

                prediction = Normal(mu, std)

            return prediction

    def energy_norm_loss(self, energy, loss_detail):
        """
        :param energy: (bs, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        energy_sq = (energy ** 2).sum(dim=-1).mean()
        energy_abs = energy.abs().sum(dim=-1).mean()
        loss_detail["energy_norm"] = energy_abs

        return energy_sq * self.energy_norm_reg_coef

    def contrastive_grad_loss(self, energy, loss_detail):
        """
        :param energy: (bs, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        if self.expand_bs:
            return 0

        nce = F.log_softmax(energy, dim=-2)[..., 0, :]      # (bs, feature_dim)
        grads_penalty = self.grad_loss(nce, loss_detail)
        
        delta_features = self.input_delta_features          # (bs, num_samples, feature_dim)
        delta_grad = torch.autograd.grad(energy.sum(), delta_features, create_graph=True)[0]

        grads_penalty += delta_grad.pow(2).mean(dim=(0, 1)).sum() * self.delta_grad_reg_coef
        loss_detail["delta_grad_norm"] = delta_grad.abs().mean(dim=0).sum()
        return grads_penalty

    def grad_loss(self, target, loss_detail):
        """
        :param target: (bs, feature_dim)
        :return:
            loss: scalar
        """
        if self.expand_bs:
            return 0

        sa_grads = torch.autograd.grad(target.sum(), self.sa_inputs, create_graph=True)

        grads_penalty = sa_grad_norm = 0
        sa_grad_reg_thre = self.grad_attn_params.sa_grad_reg_thre
        sa_grad_reg_pow = self.grad_attn_params.sa_grad_reg_pow

        for grad in sa_grads:
            # grad: (feature_dim, bs, obj_i_dim) or (feature_dim, bs, action_dim)
            grads_abs = grad.abs()
            sa_grad_norm += grads_abs.mean(dim=1).sum()

            grads_lt_thre_abs = grads_abs * (grads_abs < sa_grad_reg_thre)
            grads_ge_thre_abs = grads_abs * (grads_abs >= sa_grad_reg_thre)

            grads_penalty += grads_lt_thre_abs.pow(sa_grad_reg_pow).mean(dim=2).sum() * self.sa_grad_lt_thre_reg_coef
            grads_penalty += grads_ge_thre_abs.pow(sa_grad_reg_pow).mean(dim=2).sum() * self.sa_grad_ge_thre_reg_coef

        loss_detail["sa_grad_norm"] = sa_grad_norm
        return grads_penalty

    def mask_loss(self, loss_detail):
        """
        :return:
            loss: scalar
        """
        if self.mask_type != "input":
            return 0

        # input_mask: (feature_dim, num_objs + 1, bs, 1)
        sa_mask_norm = self.input_mask.mean(dim=2).sum()
        loss = sa_mask_norm * self.mask_reg_coef
        loss_detail["sa_mask_norm"] = sa_mask_norm

        return loss

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs:
            if self.parallel_sample and self.training: {obs_i_key: (feature_dim, bs, obs_i_shape)}
            else: {obs_i_key: (bs, obs_i_shape)}
            notice that bs can be a multi-dimensional batch size
        :param actions: 
            if self.parallel_sample and self.training: (feature_dim, bs, num_pred_steps, action_dim)
            else: (bs, num_pred_steps, action_dim)
        :param next_obses:
            if self.parallel_sample and self.training: {obs_i_key: (feature_dim, bs, obs_i_shape)}
            else: {obs_i_key: (bs, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """

        bs, num_pred_steps = actions.shape[:-2], actions.shape[-2]

        if num_pred_steps > 1:
            raise NotImplementedError

        if self.parallel_sample and self.training:
            bs = bs[1:]

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_features = self.encoder(next_obses)

        action = actions[..., 0, :]
        if self.continuous_state:
            next_feature = next_features[..., 0, :]
        else:
            next_feature = [next_feature_i[..., 0, :] for next_feature_i in next_features]

        neg_delta_feature = None
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (bs, num_negative_samples, feature_dim)
            neg_delta_feature = self.sample_delta_feature(bs, self.num_negative_samples)

        if self.mask_type == "sample":
            size = (self.feature_dim,) + bs
            if self.training:
                # (feature_dim, bs, num_objs + 1)
                self.sampled_sa_mask = get_train_mask(size, self.num_objs + 1, self.device)
            else:
                # (feature_dim, bs, num_objs + 2, num_objs + 1)
                self.sampled_sa_mask = get_eval_mask(size, self.num_objs + 1, self.device)

        prediction = self.forward_with_feature(obj_feature, feature, action, next_feature, neg_delta_feature)

        loss_detail = {}

        if not self.continuous_state:
            # if self.parallel_sample and self.training: [(feature_dim, bs, feature_i_dim)] * feature_dim
            # else: [(bs, feature_i_dim)] * feature_dim
            label = next_feature

            # [(bs,)] * feature_dim
            if self.parallel_sample and self.training:
                label = [label_i[i].argmax(dim=-1) for i, label_i in enumerate(label)]
            else:
                label = [label_i.argmax(dim=-1) for label_i in label]

            label = torch.stack(label, dim=-1)                                      # (bs, feature_dim)
            label = label.unsqueeze(dim=-1)                                         # (bs, feature_dim, 1)

            if self.expand_bs:
                label = label.unsqueeze(dim=1).expand(-1, self.num_objs + 2, -1, -1)

            logits = prediction
            log_softmax = F.log_softmax(logits, dim=-1)                             # (bs, feature_dim, final_dim)
            ce = -torch.gather(log_softmax, -1, label)[..., 0]                      # (bs, feature_dim)
            loss = ce.sum(dim=-1).mean()
            loss_detail["ce_loss"] = loss

            grad_loss = self.grad_loss(ce, loss_detail)
            global_causality_update_target = ce
        elif self.dynamics_mode == "contrastive":
            # energy: (bs, 1 + num_negative_samples, feature_dim)
            energy = prediction
            nce = F.log_softmax(energy, dim=-2)[..., 0, :]
            loss = -nce.sum(dim=-1).mean()
            loss_detail["nce_loss"] = loss
            energy_norm_loss = self.energy_norm_loss(energy, loss_detail)
            loss = loss + energy_norm_loss
            grad_loss = self.contrastive_grad_loss(energy, loss_detail)

            global_causality_update_target = energy if self.mask_type == "sample" else nce
        else:
            if self.parallel_sample and self.training:
                assert len(bs) == 1
                feature_dim = self.feature_dim
                eye = torch.eye(feature_dim, device=self.device)[:, None, :]        # (feature_dim, 1, feature_dim)
                next_feature = (next_feature * eye).sum(dim=-1).T                   # (bs, feature_dim)

            if self.expand_bs:
                next_feature = next_feature.unsqueeze(dim=1).expand(-1, self.num_objs + 2, -1)

            if self.dynamics_mode == "pred_scalar":
                mse = (prediction - next_feature).pow(2)                            # (bs, feature_dim)
                loss = mse.sum(dim=-1).mean()
                loss_detail["mse_loss"] = loss
                grad_loss = self.grad_loss(mse, loss_detail)

                global_causality_update_target = mse
            else:
                nll = -prediction.log_prob(next_feature)                            # (bs, feature_dim)
                loss = nll.sum(dim=-1).mean()
                loss_detail["nll_loss"] = loss
                grad_loss = self.grad_loss(nll, loss_detail)

                global_causality_update_target = nll

        mask_loss = self.mask_loss(loss_detail)

        loss = loss + grad_loss + mask_loss

        if eval:
            self.update_global_causality(global_causality_update_target)
        else:
            self.backprop(loss, loss_detail)

            if self.parallel_sample:
                if not self.continuous_state:
                    priority = 1 - torch.exp(-ce)                                   # (bs, feature_dim)
                elif self.dynamics_mode == "contrastive":
                    priority = 1 - nce.exp()                                        # (bs, feature_dim)
                elif self.dynamics_mode == "pred_scalar":
                    priority = mse                                                  # (bs, feature_dim)
                else:
                    priority = nll.exp()                                            # (bs, feature_dim)

                loss_detail["priority"] = priority.T

        return loss_detail

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.dynamics_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        if torch.isfinite(grad_norm):
            self.optimizer.step()

        return loss_detail

    def get_grad(self, target, mean_across_bs=True):
        # target: (bs, feature_dim)
        sa_grads = torch.autograd.grad(target.sum(), self.sa_inputs)
        obj_inner_dim = self.obj_encoder.obj_inner_dim
        obj_keys = self.obj_encoder.obj_keys

        grad_mask = []
        for i, grad in enumerate(sa_grads):
            # grad: (feature_dim, bs, action_dim / obj_i_dim)
            if mean_across_bs:
                grad_norm = grad.abs().mean(dim=1)                      # (feature_dim, bs, action_dim / obj_i_dim)
            else:
                grad_norm = grad.abs().permute(1, 0, 2)

            if not self.continuous_state:
                if i < self.num_objs:
                    # grad of object
                    obj_i_inner_dim = obj_inner_dim[obj_keys[i]]
                    grad_norm = torch.split(grad_norm, obj_i_inner_dim, dim=-1)
                    grad_norm = torch.stack([grad.sum(dim=-1) for grad in grad_norm], dim=-1)
                else:
                    # grad of action
                    grad_norm = grad_norm.sum(dim=-1, keepdim=True)

            grad_mask.append(grad_norm)

        grad_mask = torch.concat(grad_mask, dim=-1)                     # (feature_dim, feature_dim + action_dim)
        return grad_mask

    def get_cmi(self, target, mean_across_bs=True):
        if not self.continuous_state or self.dynamics_mode == "pred_normal":
            # target: (bs, num_objs + 2, feature_dim)
            mask_nll, full_nll = target[..., :-1, :], target[..., -1:, :]
            cmi = mask_nll - full_nll                           # (bs, num_objs + 1, feature_dim)
        elif self.dynamics_mode == "contrastive":
            # target: (bs, num_objs + 1, 1 + num_negative_samples, feature_dim)
            mask_energy, full_energy = target[..., :-1, :, :], target[..., -1:, :, :]
            # (bs, num_objs + 1, 1 + num_negative_samples, feature_dim)
            cond_energy = full_energy - mask_energy             
            pos_cond_energy = cond_energy[..., 0, :]            # (bs, num_objs + 1, feature_dim)

            K = mask_energy.shape[-2]                           # num_negative_samples
            neg_energy = mask_energy[..., 1:, :]                # (bs, num_objs + 1, num_negative_samples, feature_dim)
            neg_cond_energy = cond_energy[..., 1:, :]           # (bs, num_objs + 1, num_negative_samples, feature_dim)

            log_w_neg = F.log_softmax(neg_energy, dim=-2)       # (bs, num_objs + 1, num_negative_samples, feature_dim)
            # (bs, num_objs + 1, num_negative_samples, feature_dim)
            weighted_neg_cond_energy = np.log(K - 1) + log_w_neg + neg_cond_energy
            # (bs, num_objs + 1, 1 + num_negative_samples, feature_dim)
            cond_energy = torch.cat([pos_cond_energy.unsqueeze(dim=-2), weighted_neg_cond_energy], dim=-2)
            log_denominator = -np.log(K) + torch.logsumexp(cond_energy, dim=-2)         # (bs, feature_dim, feature_dim)
            cmi = pos_cond_energy - log_denominator                                     # (bs, num_objs + 1, feature_dim)
        else:
            # target: (bs, num_objs + 1, feature_dim)
            mask_error, full_error = target[..., :-1, :], target[..., -1:, :]
            cmi = mask_error - full_error

        if mean_across_bs:
            cmi = cmi.view(-1, self.num_objs + 1, self.feature_dim).mean(dim=0).T       # (feature_dim, num_objs + 1)
        else:
            if cmi.ndim > 3:
                raise NotImplementedError
            cmi = cmi.permute(0, 2, 1)                                                  # (bs, feature_dim, num_objs + 1)

        return cmi

    def update_global_causality(self, target):
        """
        :param target: (bs, feature_dim)
        :return:
        """
        if self.mask_type == "sample":
            assert self.expand_bs
            batch_causality = self.get_cmi(target)
        else:
            assert not self.expand_bs
            batch_causality = self.get_grad(target)

        eval_tau = self.grad_attn_params.eval_tau
        self.global_causality_val = eval_tau * self.global_causality_val + (1 - eval_tau) * batch_causality.detach()
        self.global_causality_mask = self.global_causality_val >= self.global_causality_threshold

    def predict_step_with_feature(self, obj_feature, feature, action, next_feature=None):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: pred_next_feature: (bs, feature_dim)
        """
        self.use_label_for_mask_eval = True
        if not self.use_label_for_mask_eval:
            raise NotImplementedError

        bs = action.shape[:-1]

        if len(bs) > 1:
            raise NotImplementedError

        next_feature_ = delta_feature_candidates = None
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (bs, num_negative_samples, feature_dim)
            delta_feature_candidates = self.sample_delta_feature(bs, self.num_pred_samples)

            if self.use_label_for_mask_eval:
                next_feature_ = next_feature

        if self.mask_type == "sample":
            size = (self.feature_dim,) + bs
            self.sampled_sa_mask = get_eval_mask(size, self.num_objs + 1, self.device)

        prediction = self.forward_with_feature(obj_feature, feature, action, next_feature_, delta_feature_candidates)

        if not self.continuous_state:
            logits = prediction                                                     # (bs, feature_dim, final_size)
            pred_next_feature = logits.argmax(dim=-1)                               # (bs, feature_dim)

            label = [label_i.argmax(dim=-1) for label_i in next_feature]            # [(bs,)] * feature_dim
            label = torch.stack(label, dim=-1)                                      # (bs, feature_dim)
            label = label.unsqueeze(dim=-1)                                         # (bs, feature_dim, 1)

            if self.expand_bs:
                label = label.unsqueeze(dim=-3).expand(-1, self.num_objs + 2, -1, -1)
                pred_next_feature = pred_next_feature[:, -1]                        # (bs, feature_dim)

            log_softmax = F.log_softmax(logits, dim=-1)                             # (bs, feature_dim, final_dim)
            ce = -torch.gather(log_softmax, -1, label)[..., 0]                      # (bs, feature_dim)

            grad_target = cmi_target = ce
        elif self.dynamics_mode == "contrastive":
            # energy: (bs, 1 + num_pred_samples, feature_dim)
            energy = prediction
            nce = F.log_softmax(energy, dim=-2)[..., 0, :]

            grad_target = nce
            cmi_target = energy

            if self.expand_bs:
                # (bs, num_objs + 2, 1 + num_pred_samples, feature_dim) -> (bs, 1 + num_pred_samples, feature_dim)
                energy = energy[..., -1, :, :]

            if self.use_label_for_mask_eval:
                energy = energy[..., 1:, :]

            argmax_idx = torch.argmax(energy, dim=-2, keepdim=True)                             # (bs, 1, feature_dim)
            delta_feature = torch.gather(delta_feature_candidates, -2, argmax_idx)[..., 0, :]   # (bs, feature_dim)
            pred_next_feature = feature + delta_feature
        else:
            if self.expand_bs:
                next_feature = next_feature.unsqueeze(dim=-2).expand(-1, self.num_objs + 2, -1)

            if self.dynamics_mode == "pred_scalar":
                pred_next_feature = prediction
                mse = (pred_next_feature - next_feature).pow(2)
                grad_target = cmi_target = mse
            else:
                pred_next_feature = prediction.mean
                nll = -prediction.log_prob(next_feature)                                        # (bs, feature_dim)
                grad_target = cmi_target = nll

            if self.expand_bs:
                pred_next_feature = pred_next_feature[..., -1, :]                               # (bs, feature_dim)

        if self.evaluate_grad_mask:
            assert not self.expand_bs
            self.grad_mask = self.get_grad(grad_target, mean_across_bs=False)

        if self.evaluate_sample_cmi_mask:
            assert self.expand_bs
            self.sample_cmi_mask = self.get_cmi(cmi_target, mean_across_bs=False)

        if self.evaluate_input_mask:
            # (feature_dim, num_objs + 1, bs, 1) -> (bs, feature_dim, num_objs + 1)
            self.input_mask = self.input_mask[..., 0].permute(2, 0, 1)

        return pred_next_feature
    
    def get_mixture_cmi_mask(self, obs, action, next_feature):
        self.use_label_for_mask_eval = True
        self.use_delta_for_samples = False

        bs = action.shape[:-1]

        if len(bs) > 1:
            raise NotImplementedError

        if not self.use_label_for_mask_eval:
            raise NotImplementedError

        if self.dynamics_mode == "contrastive" or self.mask_type == "sample":
            raise NotImplementedError

        device = self.device
        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        original_prediction = self.forward_with_feature(obj_feature, feature, action, next_feature)

        vary_predictions = []
        num_variations = self.feature_dim + (self.action_dim if self.continuous_action else 1)
        for i in range(num_variations):
            num_samples = 30
            if i < self.feature_dim and not self.continuous_state:
                num_samples = self.encoder.feature_inner_dim[i]
            elif i >= self.feature_dim and not self.continuous_action:
                num_samples = self.action_dim

            obs_i = {k: v.unsqueeze(dim=1).expand(-1, num_samples, -1) for k, v in obs.items()}
            action_i = action.unsqueeze(dim=1).expand(-1, num_samples, -1)

            self.feature = self.encoder(obs_i)

            if i < self.feature_dim:
                key_i, offset_i = self.encoder.index2key[i]
                obs_i_val_i = obs_i[key_i].clone()

                if self.continuous_state:
                    if self.use_delta_for_samples:
                        delta_i_min, delta_i_max = self.delta_feature_min[i], self.delta_feature_max[i]
                        samples = torch.rand(num_samples, device=device) * (delta_i_max - delta_i_min) + delta_i_min
                        overwrite_val = obs_i_val_i[:, :, offset_i] + samples
                    else:
                        obs_i_min, obs_i_max = -1, 1
                        overwrite_val = torch.rand(num_samples, device=device) * (obs_i_max - obs_i_min) + obs_i_min
                else:
                    overwrite_val = torch.arange(num_samples, device=device)

                obs_i_val_i[:, :, offset_i] = overwrite_val
                obs_i[key_i] = obs_i_val_i
            else:
                action_i = action_i.clone()

                if self.continuous_action:
                    act_i_min, act_i_max = -1, 1
                    overwrite_val = torch.rand(num_samples, device=device) * (act_i_max - act_i_min) + act_i_min
                else:
                    overwrite_val = torch.arange(num_samples, device=device)

                action_i[:, :, i - self.feature_dim] = overwrite_val

            feature = self.encoder(obs_i)
            obj_feature = self.obj_encoder(obs_i)
            vary_prediction = self.forward_with_feature(obj_feature, feature, action_i, next_feature)
            vary_predictions.append(vary_prediction)

        if not self.continuous_state:
            label = [label_i.argmax(dim=-1) for label_i in next_feature]    # [(bs,)] * feature_dim
            label = torch.stack(label, dim=-1)                              # (bs, feature_dim)
            label = label.unsqueeze(dim=-1)                                 # (bs, feature_dim, 1)

            logits = original_prediction                                    # (bs, feature_dim, final_size)
            log_softmax = F.log_softmax(logits, dim=-1)                     # (bs, feature_dim, final_dim)
            ori_nll = -torch.gather(log_softmax, -1, label)[..., 0]         # (bs, feature_dim)

            vary_nlls = []
            for logits in vary_predictions:
                softmax = F.softmax(logits, dim=-1)                         # (bs, num_samples, feature_dim, final_dim)
                softmax = softmax.mean(dim=-3)                              # (bs, feature_dim, final_dim)
                log_softmax = torch.log(softmax)                            # (bs, feature_dim, final_dim)
                vary_nll = -torch.gather(log_softmax, -1, label)[..., 0]    # (bs, feature_dim)
                vary_nlls.append(vary_nll)
            vary_nlls = torch.stack(vary_nlls, dim=-1)                      # (bs, feature_dim, feature_dim + action_dim)

            cmi = vary_nlls - ori_nll.unsqueeze(dim=-1)
        else:
            if self.dynamics_mode == "pred_scalar":
                original_prediction = Normal(original_prediction, torch.ones_like(original_prediction))
            ori_logp = original_prediction.log_prob(next_feature)           # (bs, feature_dim)

            vary_logps = []
            next_feature = next_feature.unsqueeze(dim=-2)
            for vary_prediction in vary_predictions:
                if self.dynamics_mode == "pred_scalar":
                    vary_prediction = Normal(vary_prediction, torch.ones_like(vary_prediction))
                vary_logp = vary_prediction.log_prob(next_feature)          # (bs, num_samples, feature_dim)
                num_samples = vary_logp.shape[-2]
                vary_logp = torch.logsumexp(vary_logp, dim=-2) / num_samples  # (bs, feature_dim)
                vary_logps.append(vary_logp)
            vary_logps = torch.stack(vary_logps, dim=-1)                    # (bs, feature_dim, feature_dim + action_dim)

            cmi = ori_logp.unsqueeze(dim=-1) - vary_logps

        self.mixture_cmi_mask = cmi
        self.feature = None

    def reset_mask_eval(self):
        self.evaluate_grad_mask = False
        self.evaluate_attn_mask = False
        self.evaluate_input_mask = False
        self.evaluate_attn_signature_mask = False
        self.evaluate_sample_cmi_mask = False
        self.evaluate_mixture_cmi_mask = False

        self.grad_mask = None
        self.attn_mask = None
        self.input_mask = None
        self.attn_signature_mask = None
        self.sample_cmi_mask = None
        self.mixture_cmi_mask = None

    def setup_mask_eval(self):
        if self.mask_type == "none":
            self.evaluate_grad_mask = True
            self.evaluate_attn_mask = True
            self.evaluate_mixture_cmi_mask = self.dynamics_mode != "contrastive"
        elif self.mask_type == "input":
            self.evaluate_grad_mask = True
            self.evaluate_attn_mask = True
            self.evaluate_mixture_cmi_mask = self.dynamics_mode != "contrastive"
            self.evaluate_input_mask = True
        elif self.mask_type == "attention":
            self.evaluate_grad_mask = True
            self.evaluate_attn_mask = True
            self.evaluate_mixture_cmi_mask = self.dynamics_mode != "contrastive"
            self.evaluate_attn_signature_mask = True
        elif self.mask_type == "sample":
            self.evaluate_sample_cmi_mask = True
        else:
            raise NotImplementedError

    def eval_prediction(self, obs, actions, next_obses):
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)

        self.setup_mask_eval()

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_features = self.encoder(next_obses)

        assert actions.ndim == 3
        num_pred_steps = actions.shape[1]
        if num_pred_steps > 1:
            raise NotImplementedError

        if self.continuous_action:
            action = actions[..., 0, :]
        else:
            action = actions[..., 0]

        if self.continuous_state:
            next_feature = next_features[..., 0, :]
        else:
            next_feature = [next_features_i[..., 0, :] for next_features_i in next_features]

        with torch.no_grad():
            if self.evaluate_mixture_cmi_mask:
                self.get_mixture_cmi_mask(obs, action, next_feature)

        pred_next_feature = self.predict_step_with_feature(obj_feature, feature, action, next_feature)
        pred_next_feature = pred_next_feature.detach()

        if not self.continuous_state:
            feature = [feature_i.argmax(dim=-1) for feature_i in feature]                       # [(bs,)] * feature_dim
            feature = torch.stack(feature, dim=-1)                                              # (bs, feature_dim)

            next_feature = [next_feature_i.argmax(dim=-1) for next_feature_i in next_feature]   # [(bs,)] * feature_dim
            next_feature = torch.stack(next_feature, dim=-1)                                    # (bs, feature_dim)

        masks = {"grad_mask": self.grad_mask,
                 "attn_mask": self.attn_mask,
                 "input_mask": self.input_mask,
                 "attn_signature_mask": self.attn_signature_mask,
                 "sample_cmi_mask": self.sample_cmi_mask,
                 "mixture_cmi_mask": self.mixture_cmi_mask}

        for k, v in masks.items():
            if v is not None:
                masks[k] = v.detach()

        self.reset_mask_eval()

        return feature, next_feature, pred_next_feature, masks

    def preprocess(self, obs, actions, next_obses):
        if isinstance(actions, int):
            actions = np.array([actions])

        if isinstance(actions, np.ndarray):
            if self.continuous_action and actions.dtype != np.float32:
                actions = actions.astype(np.float32)
            if not self.continuous_action and actions.dtype != np.int64:
                actions = actions.astype(np.int64)
            actions = torch.from_numpy(actions).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            next_obses = postprocess_obs(preprocess_obs(next_obses, self.params))
            next_obses = {k: torch.from_numpy(v).to(self.device) for k, v in next_obses.items()}

        need_squeeze = False
        if actions.ndim == 1:
            need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}                          # (bs, obs_spec)
            actions = actions[None, None]                                       # (bs, num_pred_steps, action_dim)
            next_obses = {k: v[None, None] for k, v in next_obses.items()}      # (bs, num_pred_steps, obs_spec)
        elif self.params.env_params.num_envs > 1 and actions.ndim == 2:
            need_squeeze = True
            actions = actions[:, None]                                          # (bs, num_pred_steps, action_dim)
            next_obses = {k: v[:, None] for k, v in next_obses.items()}         # (bs, num_pred_steps, obs_spec)

        return obs, actions, next_obses, need_squeeze

    def get_threshold(self):
        return self.global_causality_threshold

    def get_mask(self, bool_mask=True):
        if bool_mask:
            return self.global_causality_mask
        else:
            return self.global_causality_val

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "global_causality_val": self.global_causality_val,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("contrastive loaded", path)
            import time
            start = time.time()
            checkpoint = torch.load(path, map_location=device)
            print("loading takes", time.time() - start)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.global_causality_val = checkpoint["global_causality_val"]
            self.global_causality_mask = self.global_causality_val >= self.global_causality_threshold
