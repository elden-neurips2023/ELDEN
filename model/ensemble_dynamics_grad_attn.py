import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta

from model.gumbel import gumbel_sigmoid
from model.modules import ChannelMHAttention
from model.dynamics_utils import reset_layer, forward_network, get_train_mask, get_eval_mask, reset_layer_orth
from model.dynamics_utils import expand_helper, flatten_helper, mixup_helper

from utils.utils import to_numpy, preprocess_obs


class EnsembleDynamicsGradAttn(nn.Module):
    def __init__(self, encoder, obj_encoder, params):
        super(EnsembleDynamicsGradAttn, self).__init__()

        self.encoder = encoder
        self.obj_encoder = obj_encoder

        self.params = params
        self.dynamics_keys = params.dynamics_keys
        self.device = device = params.device
        self.dynamics_params = dynamics_params = params.dynamics_params

        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action
        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer
        self.orthogonal_initialization = dynamics_params.orthogonal_init
        self.grad_target_logit = dynamics_params.grad_target_logit
        self.new_uncertainty_calculation = dynamics_params.new_uncertainty_calculation

        self.dynamics_params = dynamics_params = params.dynamics_params
        self.num_pred_steps = dynamics_params.num_pred_steps
        if self.num_pred_steps > 1:
            raise NotImplementedError
        self.ensemble_size = dynamics_params.ensemble_size
        self.monolithic_arch = dynamics_params.monolithic_arch
        self.intrinsic_reward_type = getattr(dynamics_params, "intrinsic_reward_type", "none")
        self.intrinsic_reward_scale = getattr(dynamics_params, "intrinsic_reward_scale", 0)
        self.local_causality_metric = getattr(dynamics_params, "local_causality_metric", "grad")
        self.local_causality_threshold = getattr(dynamics_params, "local_causality_threshold", 0)
        self.use_dynamics_uncertainty_mask = getattr(dynamics_params, "use_dynamics_uncertainty_mask", False)
        self.dynamics_uncertainty_mask_threshold = getattr(dynamics_params, "dynamics_uncertainty_mask_threshold", 0)

        self.mixup = getattr(dynamics_params, "mixup", False)
        if self.continuous_state or self.continuous_action:
            self.mixup = False

        if self.mixup:
            alpha = torch.tensor(float(getattr(dynamics_params, "mixup_alpha", 0.1)), device=self.device)
            self.beta = Beta(alpha, alpha)

        if self.intrinsic_reward_type == "dynamics_curiosity":
            assert self.ensemble_size == 1
        if self.intrinsic_reward_type == "cai":
            assert self.local_causality_metric == "sample_cmi"

        if "uncertainty" in self.intrinsic_reward_type:
            assert self.ensemble_size > 1

        self.contrastive_params = contrastive_params = dynamics_params.contrastive_params
        self.num_negative_samples = contrastive_params.num_negative_samples
        self.num_pred_samples = contrastive_params.num_pred_samples
        self.num_pred_iters = contrastive_params.num_pred_iters
        self.pred_sigma_init = contrastive_params.pred_sigma_init
        self.pred_sigma_shrink = contrastive_params.pred_sigma_shrink
        self.energy_norm_reg_coef = contrastive_params.energy_norm_reg_coef
        self.delta_grad_reg_coef = contrastive_params.delta_grad_reg_coef

        # (feature_dim,)
        if self.continuous_state:
            self.delta_feature_min = self.encoder({key: val[0] for key, val in self.params.obs_delta_range.items()})
            self.delta_feature_max = self.encoder({key: val[1] for key, val in self.params.obs_delta_range.items()})

        self.grad_attn_params = params.dynamics_params.grad_attn_params

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

        if self.monolithic_arch:
            channel_shape = (self.ensemble_size, 1)
        else:
            channel_shape = (self.ensemble_size, feature_dim)

        # Instantiate the parameters of each module in each variable's dynamics model
        # state action feature extractor
        in_dim = input_dim
        for out_dim in grad_attn_params.feature_fc_dims:
            self.sa_feature_weights.append(nn.Parameter(torch.zeros(*channel_shape, num_objs + 1, in_dim, out_dim)))
            self.sa_feature_biases.append(nn.Parameter(torch.zeros(*channel_shape, num_objs + 1, 1, out_dim)))
            in_dim = out_dim

        # multi-head attention
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

        num_queries = num_keys = num_objs + 1

        self.attns = nn.ModuleList()
        in_dim = feature_embed_dim
        for i in range(grad_attn_params.num_attns):
            in_dim = attn_out_dim if i else feature_embed_dim
            attn = ChannelMHAttention(channel_shape, attn_dim, num_heads, num_queries, in_dim, num_keys, in_dim,
                                      out_dim=attn_out_dim, use_bias=use_bias, residual=residual,
                                      share_weight_across_kqv=share_weight_across_kqv, post_fc_dims=post_fc_dims)
            self.attns.append(attn)

        self.mask_params = mask_params = grad_attn_params.mask_params
        self.mask_type = mask_params.mask_type
        assert self.mask_type in ["none", "sample", "input", "attention"]

        # learnable input / attention mask
        if self.mask_type in ["input", "attention"]:

            self.learn_signature_queries = mask_params.learn_signature_queries
            if self.learn_signature_queries:
                b = 1 / np.sqrt(in_dim)
                self.signature_queries = nn.Parameter(torch.FloatTensor(*channel_shape, num_queries, in_dim).uniform_(-b, b))

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
            in_dim = feature_embed_dim
            for i in range(mask_params.num_mask_attns):
                in_dim = attn_out_dim if i else in_dim
                attn = ChannelMHAttention(channel_shape, attn_dim, num_heads, num_queries, in_dim, num_keys, in_dim,
                                          out_dim=attn_out_dim, use_bias=use_bias, residual=residual,
                                          share_weight_across_kqv=share_weight_across_kqv, post_fc_dims=post_fc_dims)
                self.mask_attns.append(attn)

            in_dim = attn_out_dim
            for out_dim in mask_params.mask_fc_dims + [mask_final_dim]:
                self.sa_mask_weights.append(nn.Parameter(torch.zeros(*channel_shape, num_objs + 1, in_dim, out_dim)))
                self.sa_mask_biases.append(nn.Parameter(torch.zeros(*channel_shape, num_objs + 1, 1, out_dim)))
                in_dim = out_dim

        self.sampled_sa_mask = None

        self.dynamics_mode = grad_attn_params.dynamics_mode
        assert self.dynamics_mode in ["contrastive", "pred_scalar", "pred_normal"]

        # energy compute
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # sa_feature encoder
            in_dim = attn_out_dim
            for out_dim in grad_attn_params.sa_encoding_fc_dims:
                self.sa_encoder_weights.append(nn.Parameter(torch.zeros(*channel_shape, in_dim, out_dim)))
                self.sa_encoder_biases.append(nn.Parameter(torch.zeros(*channel_shape, 1, out_dim)))
                in_dim = out_dim

            # delta feature encoder
            in_dim = 1
            for out_dim in grad_attn_params.delta_encoding_fc_dims:
                self.d_encoder_weights.append(nn.Parameter(torch.zeros(*channel_shape, in_dim, out_dim)))
                self.d_encoder_biases.append(nn.Parameter(torch.zeros(*channel_shape, 1, out_dim)))
                in_dim = out_dim
        # prediction compute
        else:
            # predictor
            in_dim = attn_out_dim
            for out_dim in grad_attn_params.predictor_fc_dims:
                self.predictor_weights.append(nn.Parameter(torch.zeros(*channel_shape, in_dim, out_dim)))
                self.predictor_biases.append(nn.Parameter(torch.zeros(*channel_shape, 1, out_dim)))
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

            self.predictor_weights.append(nn.Parameter(torch.zeros(*channel_shape, in_dim, final_size)))
            self.predictor_biases.append(nn.Parameter(torch.zeros(*channel_shape, 1, final_size)))

    def reset_params(self):
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
                if self.orthogonal_initialization:
                    reset_layer_orth(w, b)
                else:
                    reset_layer(w, b)

    def init_graph(self):
        feature_dim = self.feature_dim
        action_dim = self.action_dim if self.continuous_action else 1

        if self.mask_type == "sample":
            num_cols = self.num_objs + 1
        else:
            num_cols = feature_dim + action_dim

        device = self.device
        self.global_causality_threshold = self.grad_attn_params.global_causality_threshold
        self.global_causality_val = torch.ones(self.ensemble_size, feature_dim, num_cols, device=device)
        self.global_causality_val = self.global_causality_val * self.global_causality_threshold
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
        :param obj_feature: [(bs, ensemble_size, feature_dim, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param action: (bs, ensemble_size, feature_dim, action_dim)
        :return: (ensemble_size, feature_dim, num_objs + 1, bs, out_dim),
        """
        ensemble_size = self.ensemble_size
        feature_dim = self.feature_dim
        num_objs = self.num_objs
        bs = action.shape[0]

        # inputs: [(bs, ensemble_size, feature_dim, input_i_dim)] * (num_objs + 1)
        inputs, self.sa_inputs = [], []
        for input_i in obj_feature + [action]:
            input_i = input_i.detach()
            input_i.requires_grad = True
            self.sa_inputs.append(input_i)

            input_i = input_i.permute(3, 1, 2, 0)   # (input_i_dim, ensemble_size, feature_dim, bs)
            inputs.append(input_i)

        x = pad_sequence(inputs)                    # (input_dim, num_objs + 1, ensemble_size, feature_dim, bs)
        x = x.permute(2, 3, 1, 4, 0)                # (ensemble_size, feature_dim, num_objs + 1, bs, input_dim)

        sa_feature = forward_network(x, self.sa_feature_weights, self.sa_feature_biases)
        sa_feature = F.relu(sa_feature)

        return sa_feature                           # (ensemble_size, feature_dim, num_objs + 1, bs, feature_embed_dim)

    def apply_sa_mask(self, sa_feature):
        """
        :param sa_feature: (ensemble_size, feature_dim, num_objs + 1, bs, feature_embed_dim)
        :return:
            sa_feature:
                if self.sampled_sa_mask.ndim == 5:
                    (ensemble_size, feature_dim, num_objs + 1, bs, num_objs + 2, feature_embed_dim)
                else:
                    (ensemble_size, feature_dim, num_objs + 1, bs, feature_embed_dim)
            log_attn_mask:
                if self.mask_type == "attention": (ensemble_size, feature_dim, bs, num_objs + 1, num_objs + 1)
                else: None
        """
        log_attn_mask = None
        self.input_mask = None
        self.expand_bs = False
        ensemble_size, feature_dim, num_objs_p_1, bs, _ = sa_feature.shape
        if self.mask_type in ["input", "attention"]:
            sa_mask_feature = sa_feature

            if not self.training or self.sa_grad_lt_thre_reg_coef > 0:
                # detach so that partial gradient regularization doesn't penalize the mask computation
                sa_mask_feature = sa_mask_feature.detach()

            if self.learn_signature_queries:
                # (ensemble_size, feature_dim, num_objs + 1, out_dim) -> (ensemble_size, feature_dim, num_objs + 1, bs, out_dim)
                queries = self.signature_queries.unsqueeze(dim=-2).expand(-1, feature_dim, -1, bs, -1)
            else:
                queries = sa_mask_feature

            for attn in self.mask_attns:
                # (ensemble_size, feature_dim, num_objs + 1, bs, attn_out_dim)
                sa_mask_feature = attn(queries, sa_mask_feature)
                queries = sa_mask_feature

            # (ensemble_size, feature_dim, num_objs + 1, bs, 1)
            sa_mask_logit = forward_network(sa_mask_feature, self.sa_mask_weights, self.sa_mask_biases)

            if self.mask_type == "input":
                if self.training:
                    self.input_mask = gumbel_sigmoid(sa_mask_logit, tau=self.tau)
                else:
                    self.input_mask = (sa_mask_logit > 0).float()
                sa_feature = sa_feature * self.input_mask
            else:
                # (ensemble_size, feature_dim, num_objs + 1, bs, signature_dim)
                attn_signatures = F.normalize(sa_mask_logit, p=2, dim=-1)
                attn_signatures_q = attn_signatures.permute(0, 1, 3, 2, 4).reshape(ensemble_size * feature_dim * bs, num_objs_p_1, -1)
                attn_signatures_k = attn_signatures_q.transpose(-1, -2)
                # (ensemble_size * feature_dim * bs, num_objs + 1, num_objs + 1)
                dist = 1 - torch.bmm(attn_signatures_q, attn_signatures_k)

                dist = dist.reshape(ensemble_size, feature_dim, bs, num_objs_p_1, num_objs_p_1)
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
            if self.sampled_sa_mask.ndim == 4:
                # self.sampled_sa_mask: (ensemble_size, feature_dim, bs, num_objs + 1)
                # sampled_sa_mask: (ensemble_size, feature_dim, num_objs + 1, bs, 1)
                sampled_sa_mask = self.sampled_sa_mask.transpose(-1, -2).unsqueeze(dim=-1)
                sa_feature = sa_feature * sampled_sa_mask
            elif self.sampled_sa_mask.ndim == 5:
                self.expand_bs = True
                # self.sampled_sa_mask: (ensemble_size, feature_dim, bs, num_objs + 2, num_objs + 1)
                # sampled_sa_mask: (ensemble_size, feature_dim, num_objs + 1, bs, num_objs + 2, 1)
                sampled_sa_mask = self.sampled_sa_mask.permute(0, 1, 4, 2, 3).unsqueeze(dim=-1)
                # (ensemble_size, feature_dim, num_objs + 1, bs, 1, out_dim)
                sa_feature = sa_feature.unsqueeze(dim=-2)
                sa_feature = sa_feature * sampled_sa_mask
                sa_feature = sa_feature.reshape(ensemble_size, feature_dim, num_objs_p_1, bs * (num_objs_p_1 + 1), -1)
            else:
                raise NotImplementedError

        return sa_feature, log_attn_mask

    def extract_sa_encoding(self, obj_feature, action):
        """
        :param obj_feature: [(bs, ensemble_size, feature_dim, obj_i_dim)] * num_objs.
            notice that bs must be 1D
        :param action: (bs, ensemble_size, feature_dim, action_dim)
        :return: (ensemble_size, feature_dim, num_objs + 1, bs, out_dim),
        """
        ensemble_size = self.ensemble_size
        feature_dim = self.feature_dim
        num_objs = self.num_objs
        bs = action.shape[0]

        # (ensemble_size, feature_dim, num_objs + 1, bs, out_dim)
        sa_feature = self.extract_state_action_feature(obj_feature, action)

        # (ensemble_size, feature_dim, num_objs + 1, bs, out_dim), (ensemble_size, feature_dim, bs, num_objs + 1, num_objs + 1)
        sa_feature, log_attn_mask = self.apply_sa_mask(sa_feature)

        if not (self.evaluate_attn_mask or self.evaluate_attn_signature_mask):
            for attn in self.attns:
                # (ensemble_size, feature_dim, num_objs + 1, bs, attn_out_dim)
                sa_feature = attn(sa_feature, sa_feature, log_attn_mask=log_attn_mask)
        else:
            attn_mask = None
            attn_signature_mask = None

            for attn in self.attns:
                # (ensemble_size, feature_dim, num_objs + 1, bs, attn_out_dim), (bs, ensemble_size, feature_dim, num_heads, num_queries, num_keys)
                sa_feature, attn_score = attn(sa_feature, sa_feature, return_attn=True, log_attn_mask=log_attn_mask)

                if self.evaluate_attn_mask:
                    attn_score = attn_score.mean(dim=3).reshape(bs * ensemble_size * feature_dim, num_objs + 1, num_objs + 1)
                    if attn_mask is None:
                        attn_mask = attn_score
                    else:
                        attn_mask = torch.bmm(attn_score, attn_mask)

                if self.evaluate_attn_signature_mask:
                    if attn_signature_mask is None:
                        attn_signature_mask = torch.exp(log_attn_mask)
                        attn_signature_mask = attn_signature_mask.reshape(ensemble_size * feature_dim * bs, num_objs + 1, num_objs + 1)
                    else:
                        attn_signature_mask = torch.bmm(attn_signature_mask, attn_signature_mask)

            if self.evaluate_attn_mask:
                attn_mask = attn_mask.reshape(bs, ensemble_size, feature_dim, num_objs + 1, num_objs + 1)
                self.attn_mask = attn_mask[:, :, np.arange(self.feature_dim), self.query_index]

            if self.evaluate_attn_signature_mask:
                attn_signature_mask = attn_signature_mask.reshape(ensemble_size, feature_dim, bs, num_objs + 1, num_objs + 1)
                attn_signature_mask = attn_signature_mask.permute(2, 0, 1, 3, 4)
                self.attn_signature_mask = attn_signature_mask[:, :, np.arange(self.feature_dim), self.query_index]

        # (ensemble_size, feature_dim, bs, out_dim)
        sa_feature = sa_feature[:, np.arange(feature_dim), self.query_index]

        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (ensemble_size, feature_dim, bs, encoding_dim)
            sa_encoding = forward_network(sa_feature, self.sa_encoder_weights, self.sa_encoder_biases)
        else:
            # (ensemble_size, feature_dim, bs, 1 / 2)
            sa_encoding = forward_network(sa_feature, self.predictor_weights, self.predictor_biases)

        if self.expand_bs:
            sa_encoding = sa_encoding.reshape(ensemble_size, feature_dim, bs, num_objs + 2, -1)

        return sa_encoding

    def forward_step(self, obj_feature, action, delta_features=None):
        """
        :param obj_feature: [(bs, ensemble_size, feature_dim, obj_i_dim)] * num_objs
            notice that bs must be 1D
        :param action: (bs, ensemble_size, feature_dim, action_dim)
        :param delta_features: (bs, ensemble_size, num_samples, feature_dim)
        """

        # extract encodings from the action & state variables
        sa_encoding = self.extract_sa_encoding(obj_feature, action)

        if self.dynamics_mode != "contrastive":
            if self.expand_bs:
                """
                sa_encoding: (ensemble_size, feature_dim, bs, num_objs + 2, -1)
                prediction:
                    if not self.continuous_state: (bs, ensemble_size, num_objs + 2, feature_dim, final_dim)
                    elif self.dynamics_mode == "pred_scalar": (bs, ensemble_size, num_objs + 2, feature_dim, 1)
                    elif self.dynamics_mode == "pred_normal": (bs, ensemble_size, num_objs + 2, feature_dim, 2)
                """
                prediction = sa_encoding.permute(2, 0, 3, 1, 4)
            else:
                """
                sa_encoding: (ensemble_size, feature_dim, bs, -1)
                prediction:
                    if not self.continuous_state: (bs, ensemble_size, feature_dim, final_dim)
                    elif self.dynamics_mode == "pred_scalar": (bs, ensemble_size, feature_dim, 1)
                    elif self.dynamics_mode == "pred_normal": (bs, ensemble_size, feature_dim, 2)
                """
                prediction = sa_encoding.permute(2, 0, 1, 3)
            return prediction

        delta_features = delta_features.detach()
        delta_features.requires_grad = True
        self.input_delta_features = delta_features

        bs, ensemble_size, num_samples, feature_dim = delta_features.shape
        delta_features = delta_features.permute(1, 3, 0, 2).reshape(ensemble_size, feature_dim, -1).unsqueeze(dim=-1)
        # (ensemble_size, feature_dim, bs, num_samples, encoding_dim)
        delta_encoding = forward_network(delta_features, self.d_encoder_weights, self.d_encoder_biases)
        delta_encoding = delta_encoding.reshape(ensemble_size, feature_dim, bs, num_samples, -1)

        if self.expand_bs:
            sa_encoding = sa_encoding.unsqueeze(dim=-2)             # (ensemble_size, feature_dim, bs, num_objs + 2, 1, encoding_dim)
            delta_encoding = delta_encoding.unsqueeze(dim=-3)       # (ensemble_size, feature_dim, bs, 1, num_samples, encoding_dim)
            energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (ensemble_size, feature_dim, bs, num_objs + 2, num_samples)
            energy = energy.permute(2, 0, 3, 4, 1)                  # (bs, ensemble_size, num_objs + 2, num_samples, feature_dim)
        else:
            sa_encoding = sa_encoding.unsqueeze(dim=-2)             # (ensemble_size, feature_dim, bs, 1, encoding_dim)
            energy = (sa_encoding * delta_encoding).sum(dim=-1)     # (ensemble_size, feature_dim, bs, num_samples)
            energy = energy.permute(2, 0, 3, 1)                     # (bs, ensemble_size, num_samples, feature_dim)

        return energy

    def preprocess_inputs(self, obj_feature, feature, action, next_feature=None, delta_features=None):
        """
        :param obj_feature:
            [(bs, obj_i_dim) or
             (bs, feature_dim, obj_i_dim) or
             (bs, ensemble_size, obj_i_dim) or
             (bs, ensemble_size, feature_dim, obj_i_dim)
            ] * num_objs
            bs can be multi-dimensional
        :param feature, action, next_feature:
            (bs, input_dim) or
            (bs, feature_dim, input_dim) or
            (bs, ensemble_size, input_dim) or
            (bs, ensemble_size, feature_dim, input_dim)
        :param delta_features: (bs, ensemble_size, num_samples, feature_dim)
        :return:
            obj_feature: [(bs, ensemble_size, feature_dim, obj_i_dim)] * num_objs
            notice that bs is 1D
            feature, action, next_feature: (bs, ensemble_size, feature_dim, input_dim)
            delta_features: (bs, ensemble_size, num_samples, feature_dim)
        """
        if not self.continuous_action and action.shape[-1] == 1:
            action = F.one_hot(action.squeeze(dim=-1), self.action_dim).float()

        input_ndim, input_shape = action.ndim, action.shape

        assert input_ndim == obj_feature[0].ndim
        assert input_ndim == (feature.ndim if self.continuous_state else feature[0].ndim)
        if next_feature is not None:
            assert input_ndim == (next_feature.ndim if self.continuous_state else next_feature[0].ndim)

        ensemble_size, feature_dim, action_dim = self.ensemble_size, self.feature_dim, self.action_dim

        self.expand_ensemble_dim, self.expand_feature_dim = True, True
        assert feature_dim != ensemble_size

        # determine whether need to expand the input
        if input_ndim == 3:
            if input_shape[-2] == ensemble_size:
                self.expand_ensemble_dim = False
            if input_shape[-2] == feature_dim:
                self.expand_feature_dim = False
        elif input_ndim == 4:
            if input_shape[-2] == feature_dim:
                self.expand_feature_dim = False
                if input_shape[-3] == ensemble_size:
                    self.expand_ensemble_dim = False
            elif input_shape[-2] == ensemble_size:
                self.expand_ensemble_dim = False

        if self.expand_feature_dim:
            obj_feature, feature, action, next_feature = \
                expand_helper([obj_feature, feature, action, next_feature], -2, feature_dim)

        if self.expand_ensemble_dim:
            obj_feature, feature, action, next_feature = \
                expand_helper([obj_feature, feature, action, next_feature], -3, ensemble_size)

        # compress multi-dimensional bs to 1-d bs
        bs = action.shape[:-3]
        obj_feature, feature, action, next_feature, delta_features = \
            flatten_helper([obj_feature, feature, action, next_feature, delta_features], -3)

        # feature and next_feature is not used as network inputs and thus do not need the feature dim expansion
        if self.continuous_state:
            feature = feature[..., np.arange(feature_dim), np.arange(feature_dim)]
            if next_feature is not None:
                next_feature = next_feature[..., np.arange(feature_dim), np.arange(feature_dim)]
        elif next_feature is not None:
            next_feature = [next_feature_i[..., i, :] for i, next_feature_i in enumerate(next_feature)]

        if self.mixup and self.training:
            num_samples = action.shape[0]
            mix_weight = self.beta.sample((num_samples, 1, 1, 1))
            rand_idx = torch.randperm(num_samples)
            action, obj_feature = mixup_helper([action, obj_feature], mix_weight, rand_idx)
            next_feature = mixup_helper(next_feature, mix_weight[..., 0], rand_idx)

        return bs, obj_feature, feature, action, next_feature, delta_features

    def forward(self, obj_feature, feature, action, next_feature=None, delta_features=None):
        """
        :param obj_feature:
            [(bs, obj_i_dim) or
             (bs, feature_dim, obj_i_dim) or
             (bs, ensemble_size, obj_i_dim) or
             (bs, ensemble_size, feature_dim, obj_i_dim)
            ] * num_objs
            bs can be multi-dimensional
        :param feature, action, next_feature:
            (bs, input_dim) or
            (bs, feature_dim, input_dim) or
            (bs, ensemble_size, input_dim) or
            (bs, ensemble_size, feature_dim, input_dim)
        :param delta_features: (bs, ensemble_size, num_samples, feature_dim)
        :return
            prediction:
                if not self.continuous_state: (bs, ensemble_size, feature_dim, final_size)
                elif self.dynamics_mode == "contrastive": (bs, ensemble_size, num_samples / 1 + num_samples, feature_dim)
                elif self.dynamics_mode == "pred_scalar": (bs, ensemble_size, feature_dim)
                elif self.dynamics_mode == "pred_normal": Normal(bs, ensemble_size, feature_dim)
            next_feature:
                if self.continuous_state: (bs, ensemble_size, feature_dim)
                else: [(bs, ensemble_size, feature_i_dim)] * feature_dim
        """
        bs, obj_feature, feature, action, next_feature, delta_features = \
            self.preprocess_inputs(obj_feature, feature, action, next_feature, delta_features)

        flatten_bs = len(bs) > 1
        if flatten_bs:
            assert not self.training
            assert self.sampled_sa_mask is None or self.sampled_sa_mask.ndim == 4

        if not self.continuous_state:
            logits = self.forward_step(obj_feature, action)             # (bs, ensemble_size, feature_dim, final_size)
            logits[..., ~self.logits_mask] = -np.inf

            if flatten_bs:
                logits = logits.reshape(*bs, *logits.shape[-3:])

            prediction = logits
        else:
            if self.dynamics_mode == "contrastive":
                assert delta_features is not None

                if next_feature is not None:
                    delta_feature = (next_feature - feature).detach()   # (bs, ensemble_size, feature_dim)

                    delta_feature = delta_feature.unsqueeze(dim=-2)     # (bs, ensemble_size, 1, feature_dim)
                    # (bs, ensemble_size, 1 + num_negative_samples, feature_dim)
                    delta_features = torch.cat([delta_feature, delta_features], dim=-2)
                else:
                    delta_features = delta_features

                # (bs, ensemble_size, 1 + num_negative_samples, feature_dim)
                energy = self.forward_step(obj_feature, action, delta_features)

                if flatten_bs:
                    energy = energy.reshape(*bs, *energy.shape[-3:])

                prediction = energy
            else:
                # if self.dynamics_mode == "pred_scalar": (bs, ensemble_size, feature_dim, 1)
                # if self.dynamics_mode == "pred_normal": (bs, ensemble_size, feature_dim, 2)
                prediction = self.forward_step(obj_feature, action)

                if self.expand_bs:
                    feature = feature.unsqueeze(dim=2).expand(-1, -1, self.num_objs + 2, -1)

                if self.dynamics_mode == "pred_scalar":
                    prediction = prediction[..., 0]                     # (bs, ensemble_size, feature_dim)
                    prediction = prediction + feature

                    if flatten_bs:
                        prediction = prediction.reshape(*bs, *prediction.shape[-2:])
                else:
                    mu, log_std = prediction.unbind(dim=-1)
                    mu = mu + feature
                    std = F.softplus(log_std).clamp(min=1e-6)

                    if flatten_bs:
                        mu = mu.reshape(*bs, *mu.shape[-2:])
                        std = std.reshape(*bs, *std.shape[-2:])

                    prediction = Normal(mu, std)

        if next_feature is not None:
            return prediction, next_feature, delta_features
        else:
            return prediction

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs: 
            {obs_i_key: (bs, obs_i_shape) or 
                        (bs, feature_dim, obs_i_shape) or 
                        (bs, ensemble_size, obs_i_shape) or 
                        (bs, ensemble_size, feature_dim, obs_i_shape)}
            notice that bs must be 1D
        :param actions:
            (bs, num_pred_steps, action_dim) or
            (bs, feature_dim, num_pred_steps, action_dim) or
            (bs, ensemble_size, num_pred_steps, action_dim) or
            (bs, ensemble_size, feature_dim, num_pred_steps, action_dim)
        :param next_obses:
            {obs_i_key: (bs, num_pred_steps, obs_i_shape) or 
                        (bs, feature_dim, num_pred_steps, obs_i_shape) or 
                        (bs, ensemble_size, num_pred_steps, obs_i_shape) or 
                        (bs, ensemble_size, feature_dim, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[0], actions.shape[-2]

        if num_pred_steps > 1:
            raise NotImplementedError

        action = actions[..., 0, :]
        next_obs = {k: v[..., 0, :] for k, v in next_obses.items()}

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_feature = self.encoder(next_obs)

        ensemble_size, feature_dim = self.ensemble_size, self.feature_dim

        # get negative samples for contrastive learning
        neg_delta_features = None
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (bs, num_negative_samples, feature_dim)
            neg_delta_features = self.sample_delta_feature([bs, ensemble_size], self.num_negative_samples)

        # get input mask for all-but-one CMI computation
        if self.mask_type == "sample":
            size = (ensemble_size, feature_dim, bs)
            if self.ensemble_size == 1 and eval:
                # (ensemble_size, feature_dim, bs, num_objs + 2, num_objs + 1)
                self.sampled_sa_mask = get_eval_mask(size, self.num_objs + 1, self.device)
            else:
                # (ensemble_size, feature_dim, bs, num_objs + 1)
                self.sampled_sa_mask = get_train_mask(size, self.num_objs + 1, self.device)

        with torch.set_grad_enabled(not (self.mask_type == "sample" and eval)):
            prediction, next_feature, delta_features = \
                self.forward(obj_feature, feature, action, next_feature, neg_delta_features)

            pred_loss, priority, grad_target, cmi_target, _ = \
                self.compute_pred_loss_and_causal_target(feature, next_feature, prediction, delta_features)

            loss_detail = {"pred_loss": pred_loss}
            if self.continuous_state and self.dynamics_mode == "contrastive":
                grad_loss = self.contrastive_grad_loss(cmi_target, loss_detail)
                energy_norm_loss = self.energy_norm_loss(cmi_target, loss_detail)
            elif self.mask_type == "sample":
                grad_loss = energy_norm_loss = 0
            else:
                grad_loss = self.grad_loss(grad_target, loss_detail)
                energy_norm_loss = 0
            mask_loss = self.mask_loss(loss_detail)

            loss = pred_loss + grad_loss + energy_norm_loss + mask_loss

            if eval:
                if self.ensemble_size == 1:
                    self.update_global_causality(grad_target, cmi_target)
            else:
                self.backprop(loss, loss_detail)

                if self.use_prioritized_buffer:
                    if self.expand_feature_dim:
                        priority = priority.mean(dim=-1)

                    if self.expand_ensemble_dim:
                        priority = priority.mean(dim=-1 if self.expand_feature_dim else -2)

                    loss_detail["priority"] = priority

        return loss_detail

    def compute_pred_loss_and_causal_target(self, feature, next_feature, prediction, delta_features=None):
        if self.expand_bs:
            next_feature = expand_helper(next_feature, -2, self.num_objs + 2)

        if not self.continuous_state:
            label = next_feature                                        # [(bs, ensemble_size, feature_i_dim)] * feature_dim
            label = [label_i.moveaxis(-1, 0) for label_i in label]      # [(feature_i_dim, bs, ensemble_size)] * feature_dim
            label = pad_sequence(label)                                 # (final_dim, feature_dim, bs, ensemble_size)
            label = label.moveaxis((0, 1), (-1, -2))                    # (bs, ensemble_size, feature_dim, final_dim)

            logits = prediction                                         # (bs, ensemble_size, feature_dim, final_dim)
            log_softmax = F.log_softmax(logits, dim=-1)                 # (bs, ensemble_size, feature_dim, final_dim)
            log_softmax = log_softmax.where(self.logits_mask, 0)
            ce = -(label * log_softmax).sum(dim=-1)                     # (bs, ensemble_size, feature_dim)

            pred_loss = ce.sum(dim=(-2, -1)).mean()
            pred_next_feature = logits.argmax(dim=-1)                   # (bs, ensemble_size, feature_dim)

            priority = 1 - torch.exp(-ce)
            if self.grad_target_logit:
                logits = logits.where(self.logits_mask, 0)
                grad_target = (label * logits).sum(dim=-1)
                cmi_target = ce
            else:
                grad_target = cmi_target = ce
        elif self.dynamics_mode == "contrastive":
            # energy: (bs, ensemble_size, 1 + num_negative_samples, feature_dim)
            energy = prediction
            nce = F.log_softmax(energy, dim=-2)[..., 0, :]                              # (bs, ensemble_size, feature_dim)
            pred_loss = -nce.sum(dim=(-2, -1)).mean()

            argmax_idx = torch.argmax(energy, dim=-2, keepdim=True)                     # (bs, ensemble_size, 1, feature_dim)
            if self.expand_bs:
                delta_features = expand_helper(delta_features, -3, self.num_objs + 2)
                feature = feature.unsqueeze(dim=-2)

            delta_feature = torch.gather(delta_features, -2, argmax_idx)[..., 0, :]     # (bs, ensemble_size, feature_dim)
            pred_next_feature = feature + delta_feature

            priority = 1 - nce.exp()
            grad_target = nce
            cmi_target = energy
        else:
            if self.dynamics_mode == "pred_scalar":
                mse = (prediction - next_feature).pow(2)                                # (bs, ensemble_size, feature_dim)
                pred_loss = mse.sum(dim=(-2, -1)).mean()
                pred_next_feature = prediction                                          # (bs, ensemble_size, feature_dim)

                if self.grad_target_logit:
                    grad_target = prediction
                    priority = cmi_target = mse
                else:
                    priority = grad_target = cmi_target = mse
            else:
                nll = -prediction.log_prob(next_feature)                                # (bs, ensemble_size, feature_dim)
                pred_loss = nll.sum(dim=(-2, -1)).mean()
                pred_next_feature = prediction.mean

                priority = None
                grad_target = cmi_target = nll

        if self.expand_bs:
            pred_next_feature = pred_next_feature[..., -1, :]

        return pred_loss, priority, grad_target, cmi_target, pred_next_feature

    def energy_norm_loss(self, energy, loss_detail):
        """
        :param energy: (bs, ensemble_size, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        energy_sq = (energy ** 2).sum(dim=-1).mean()
        energy_abs = energy.abs().sum(dim=-1).mean()
        loss_detail["energy_norm"] = energy_abs

        return energy_sq * self.energy_norm_reg_coef

    def contrastive_grad_loss(self, energy, loss_detail):
        """
        :param energy: (bs, ensemble_size, 1 + num_negative_samples, feature_dim)
        :return:
            loss: scalar
        """
        if self.expand_bs:
            return 0

        nce = F.log_softmax(energy, dim=-2)[..., 0, :]      # (bs, ensemble_size, feature_dim)
        grads_penalty = self.grad_loss(nce, loss_detail)
        
        delta_features = self.input_delta_features          # (bs, ensemble_size, num_samples, feature_dim)
        delta_grad = torch.autograd.grad(energy.sum(), delta_features, create_graph=True)[0]

        grads_penalty += delta_grad.pow(2).mean(dim=(0, 1)).sum() * self.delta_grad_reg_coef
        loss_detail["delta_grad_norm"] = delta_grad.abs().mean(dim=0).sum()
        return grads_penalty

    def grad_loss(self, target, loss_detail):
        """
        :param target: (bs, ensemble_size, feature_dim)
        :return:
            loss: scalar
        """
        if self.expand_bs:
            return 0

        if self.training and not (self.sa_grad_lt_thre_reg_coef or self.sa_grad_ge_thre_reg_coef):
            return 0

        sa_grads = torch.autograd.grad(target.sum(), self.sa_inputs, create_graph=True)

        grads_penalty = sa_grad_norm = 0
        sa_grad_reg_thre = self.grad_attn_params.sa_grad_reg_thre
        sa_grad_reg_pow = self.grad_attn_params.sa_grad_reg_pow

        for grad in sa_grads:
            # grad: (bs, ensemble_size, feature_dim, obj_i_dim or action_dim)
            grads_abs = grad.abs()
            sa_grad_norm += grads_abs.mean(dim=0).sum()

            grads_lt_thre_abs = grads_abs * (grads_abs < sa_grad_reg_thre)
            grads_ge_thre_abs = grads_abs * (grads_abs >= sa_grad_reg_thre)

            grads_penalty += grads_lt_thre_abs.pow(sa_grad_reg_pow).mean(dim=0).sum() * self.sa_grad_lt_thre_reg_coef
            grads_penalty += grads_ge_thre_abs.pow(sa_grad_reg_pow).mean(dim=0).sum() * self.sa_grad_ge_thre_reg_coef

        loss_detail["sa_grad_norm"] = sa_grad_norm
        return grads_penalty

    def mask_loss(self, loss_detail):
        """
        :return:
            loss: scalar
        """
        if self.mask_type != "input":
            return 0

        # input_mask: (ensemble_size, feature_dim, num_objs + 1, bs, 1)
        sa_mask_norm = self.input_mask.mean(dim=-2).sum()
        loss = sa_mask_norm * self.mask_reg_coef
        loss_detail["sa_mask_norm"] = sa_mask_norm

        return loss

    def backprop(self, loss, loss_detail):
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = self.dynamics_params.grad_clip_norm
        if not grad_clip_norm:
            grad_clip_norm = np.inf
        loss_detail["grad_norm"] = grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)

        if torch.isfinite(loss) and torch.isfinite(grad_norm):
            self.optimizer.step()

        return loss_detail

    def get_grad(self, target, mean_across_bs=True, aggregate_discrete=True):
        # target: (bs, ensemble_size, feature_dim)
        sa_grads = torch.autograd.grad(target.sum(), self.sa_inputs)
        obj_inner_dim = self.obj_encoder.obj_inner_dim
        obj_keys = self.obj_encoder.obj_keys

        grad_mask = []
        for i, (grad, sa_input) in enumerate(zip(sa_grads, self.sa_inputs)):
            # grad: (bs, ensemble_size, feature_dim, action_dim / obj_i_dim)
            grad = grad.abs()

            if aggregate_discrete:
                if i < self.num_objs and not self.continuous_state:
                    # grad of object
                    grad = grad * sa_input
                    obj_i_inner_dim = obj_inner_dim[obj_keys[i]]
                    grad = torch.split(grad, obj_i_inner_dim, dim=-1)
                    grad = torch.stack([grad_.sum(dim=-1) for grad_ in grad], dim=-1)
                elif i == self.num_objs and not self.continuous_action:
                    # grad of action
                    grad = grad * sa_input
                    grad = grad.sum(dim=-1, keepdim=True)

            if mean_across_bs:
                grad = grad.mean(dim=0)                         # (ensemble_size, feature_dim, action_dim / obj_i_dim)

            grad_mask.append(grad)

        grad_mask = torch.concat(grad_mask, dim=-1)             # (ensemble_size, feature_dim, feature_dim + action_dim)
        return grad_mask

    def get_cmi(self, target, mean_across_bs=True):
        if not self.continuous_state or self.dynamics_mode == "pred_normal":
            # target: (bs, ensemble_size, num_objs + 2, feature_dim)
            mask_nll, full_nll = target[..., :-1, :], target[..., -1:, :]
            cmi = mask_nll - full_nll                           # (bs, ensemble_size, num_objs + 1, feature_dim)
        elif self.dynamics_mode == "contrastive":
            # target: (bs, ensemble_size, num_objs + 2, 1 + num_negative_samples, feature_dim)
            mask_energy, full_energy = target[..., :-1, :, :], target[..., -1:, :, :]
            # (bs, ensemble_size, num_objs + 1, 1 + num_negative_samples, feature_dim)
            cond_energy = full_energy - mask_energy             
            pos_cond_energy = cond_energy[..., 0, :]            # (bs, ensemble_size, num_objs + 1, feature_dim)

            K = mask_energy.shape[-2]                           # num_negative_samples
            neg_energy = mask_energy[..., 1:, :]                # (bs, ensemble_size, num_objs + 1, num_negative_samples, feature_dim)
            neg_cond_energy = cond_energy[..., 1:, :]           # (bs, ensemble_size, num_objs + 1, num_negative_samples, feature_dim)

            log_w_neg = F.log_softmax(neg_energy, dim=-2)       # (bs, ensemble_size, num_objs + 1, num_negative_samples, feature_dim)
            # (bs, ensemble_size, num_objs + 1, num_negative_samples, feature_dim)
            weighted_neg_cond_energy = np.log(K - 1) + log_w_neg + neg_cond_energy
            # (bs, ensemble_size, num_objs + 1, 1 + num_negative_samples, feature_dim)
            cond_energy = torch.cat([pos_cond_energy.unsqueeze(dim=-2), weighted_neg_cond_energy], dim=-2)
            log_denominator = -np.log(K) + torch.logsumexp(cond_energy, dim=-2)         # (bs, ensemble_size, feature_dim, feature_dim)
            cmi = pos_cond_energy - log_denominator                                     # (bs, ensemble_size, num_objs + 1, feature_dim)
        else:
            # target: (bs, ensemble_size, num_objs + 2, feature_dim)
            mask_error, full_error = target[..., :-1, :], target[..., -1:, :]
            cmi = mask_error - full_error

        cmi = cmi.transpose(-2, -1)                                                     # (bs, ensemble_size, feature_dim, num_objs + 1)
        if mean_across_bs:
            cmi = cmi.reshape(-1, self.ensemble_size, self.feature_dim, self.num_objs + 1).mean(dim=0)

        return cmi

    def update_global_causality(self, grad_target, cmi_target):
        if self.mask_type == "sample":
            assert self.expand_bs
            batch_causality = self.get_cmi(cmi_target)
        else:
            assert not self.expand_bs
            batch_causality = self.get_grad(grad_target)

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
        ensemble_size, feature_dim = self.ensemble_size, self.feature_dim

        delta_feature_candidates = None
        if self.continuous_state and self.dynamics_mode == "contrastive":
            # (ensemble_size, bs, num_pred_samples, feature_dim)
            delta_feature_candidates = self.sample_delta_feature(bs + [ensemble_size], self.num_pred_samples)

            if not self.use_label_for_mask_eval:
                next_feature = None

        if self.evaluate_sample_cmi_mask:
            assert self.mask_type == "sample" and len(bs) == 1
            self.sampled_sa_mask = get_eval_mask((ensemble_size, feature_dim,) + bs, self.num_objs + 1, self.device)

        prediction, next_feature, delta_features = \
            self.forward(obj_feature, feature, action, next_feature, delta_feature_candidates)

        _, self.priority, grad_target, cmi_target, pred_next_feature = \
            self.compute_pred_loss_and_causal_target(feature, next_feature, prediction, delta_features)

        if self.evaluate_grad_mask:
            assert not self.expand_bs
            self.grad_mask = self.get_grad(grad_target, mean_across_bs=False)

        if self.evaluate_sample_cmi_mask:
            assert self.expand_bs
            self.sample_cmi_mask = self.get_cmi(cmi_target, mean_across_bs=False)

        if self.evaluate_input_mask:
            # (ensemble_size, feature_dim, num_objs + 1, bs, 1) -> (bs, ensemble_size, feature_dim, num_objs + 1)
            self.input_mask = self.input_mask[..., 0].permute(3, 0, 1, 2)
            self.input_mask = self.input_mask.view(*bs, *self.input_mask.shape[-3:])

        return pred_next_feature
    
    def get_mixture_cmi_mask(self, obs, action, next_feature):
        self.use_label_for_mask_eval = True
        self.use_delta_for_samples = False
        self.num_mixture_samples = 30

        bs = action.shape[:-1]

        if not self.use_label_for_mask_eval:
            raise NotImplementedError

        if self.dynamics_mode == "contrastive" or self.mask_type == "sample":
            raise NotImplementedError

        device = self.device
        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        orig_prediction, next_feature, _ = self.forward(obj_feature, feature, action, next_feature)

        # modify obs / action to generate all-but-changed-one predictions
        vary_predictions = []
        num_variations = self.feature_dim + (self.action_dim if self.continuous_action else 1)
        for i in range(num_variations):
            if not self.continuous_state and i < self.feature_dim:
                num_samples = self.encoder.feature_inner_dim[i]
            elif not self.continuous_action and i >= self.feature_dim:
                num_samples = self.action_dim
            else:
                num_samples = self.num_mixture_samples

            obs_i = expand_helper(obs, -2, num_samples)
            action_i = expand_helper(action, -2, num_samples)

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
                i = i - self.feature_dim
                action_i = action_i.clone()

                if self.continuous_action:
                    act_low, act_high = self.params.action_spec
                    act_i_min, act_i_max = act_low[i], act_high[i]
                    overwrite_val = torch.rand(num_samples, device=device) * (act_i_max - act_i_min) + act_i_min
                else:
                    overwrite_val = torch.arange(num_samples, device=device)

                action_i[:, :, i] = overwrite_val

            obj_feature_i = self.obj_encoder(obs_i)
            feature_i = expand_helper(feature, -2, num_samples)                 # (bs, num_samples, feature_dim)
            vary_prediction = self.forward(obj_feature_i, feature_i, action_i)
            vary_predictions.append(vary_prediction)

        if not self.continuous_state:
            label = [label_i.argmax(dim=-1) for label_i in next_feature]        # [(bs, ensemble_size)] * feature_dim
            label = torch.stack(label, dim=-1)                                  # (bs, ensemble_size, feature_dim)
            label = label.unsqueeze(dim=-1)                                     # (bs, ensemble_size, feature_dim, 1)

            logits = orig_prediction                                            # (bs, ensemble_size, feature_dim, final_size)
            log_softmax = F.log_softmax(logits, dim=-1)                         # (bs, ensemble_size, feature_dim, final_dim)
            ori_nll = -torch.gather(log_softmax, -1, label)[..., 0]             # (bs, ensemble_size, feature_dim)

            vary_nlls = []
            for logits in vary_predictions:
                softmax = F.softmax(logits, dim=-1)                             # (bs, num_samples, ensemble_size, feature_dim, final_dim)
                softmax = softmax.mean(dim=-4)                                  # (bs, ensemble_size, feature_dim, final_dim)
                log_softmax = torch.log(softmax)                                # (bs, ensemble_size, feature_dim, final_dim)
                vary_nll = -torch.gather(log_softmax, -1, label)[..., 0]        # (bs, ensemble_size, feature_dim)
                vary_nlls.append(vary_nll)
            vary_nlls = torch.stack(vary_nlls, dim=-1)                          # (bs, ensemble_size, feature_dim, feature_dim + action_dim)

            cmi = vary_nlls - ori_nll.unsqueeze(dim=-1)
        else:
            if self.dynamics_mode == "pred_scalar":
                orig_prediction = Normal(orig_prediction, torch.ones_like(orig_prediction))

            ori_logp = orig_prediction.log_prob(next_feature)                   # (bs, ensemble_size, feature_dim)

            vary_logps = []
            next_feature = next_feature.unsqueeze(dim=-3)                       # (bs, 1, ensemble_size, feature_dim)
            for vary_prediction in vary_predictions:

                if self.dynamics_mode == "pred_scalar":
                    vary_prediction = Normal(vary_prediction, torch.ones_like(vary_prediction))

                vary_logp = vary_prediction.log_prob(next_feature)              # (bs, num_samples, ensemble_size, feature_dim)
                num_samples = vary_logp.shape[-2]
                vary_logp = torch.logsumexp(vary_logp, dim=-2) / num_samples    # (bs, ensemble_size, feature_dim)
                vary_logps.append(vary_logp)
            vary_logps = torch.stack(vary_logps, dim=-1)                        # (bs, ensemble_size, feature_dim, feature_dim + action_dim)

            cmi = ori_logp.unsqueeze(dim=-1) - vary_logps

        self.mixture_cmi_mask = cmi
        self.feature = None
    
    def get_loss_grad_landscape(self, obs, action, next_obs, vary_key, vary_offset, num_samples=50):
        obs, action, next_obs = self.preprocess(obs, action, next_obs)

        assert action.ndim == 2

        action_ohe = F.one_hot(action, self.action_dim).float()
        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_feature = self.encoder(next_obs)

        feature = expand_helper(feature, 0, num_samples)
        next_feature = expand_helper(next_feature, 0, num_samples)

        lin = torch.linspace(0, 1, num_samples, device=self.device)[:, None, None]

        if vary_key in obs:
            dim = self.params.obs_dims[vary_key][vary_offset]
        else:
            assert vary_key == "action"
            dim = self.action_dim

        losses, grads = [],  []
        for i in range(dim):
            obs_i, action_i = obs, action
            if vary_key in obs:
                obs_i = {k: v.detach().clone() for k, v in obs.items()}
                obs_i[vary_key][..., vary_offset] = i
            else:
                action_i = action.detach().clone()
                action_i[..., :] = i
            
            obj_feature_i = self.obj_encoder(obs_i)
            action_i = F.one_hot(action_i, self.action_dim).float()

            obj_feature_i = [v1 * (1 - lin) + v2 * lin for v1, v2 in zip(obj_feature, obj_feature_i)]
            action_i = action_ohe * (1 - lin) + action_i * lin

            logits, next_feature_, _ = self.forward(obj_feature_i, feature, action_i, next_feature)

            label = [label_i.argmax(dim=-1) for label_i in next_feature_]       # [(bs, ensemble_size)] * feature_dim
            label = torch.stack(label, dim=-1)                                  # (bs, ensemble_size, feature_dim)
            label = label.unsqueeze(dim=-1)                                     # (bs, ensemble_size, feature_dim, 1)

            log_softmax = F.log_softmax(logits, dim=-1)                         # (num_samples, bs, ensemble_size, feature_dim, final_dim)
            ce = -torch.gather(log_softmax, -1, label)[..., 0]                  # (num_samples, bs, ensemble_size, feature_dim)
            grad = self.get_grad(ce, mean_across_bs=False, aggregate_discrete=False)    # (num_samples, bs, ensemble_size, feature_dim, feature_dim + action_dim)
            losses.append(to_numpy(ce[:, 0]))
            grads.append(to_numpy(grad[:, 0]))

        return losses, grads

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
            self.evaluate_mixture_cmi_mask = False
        elif self.mask_type == "input":
            self.evaluate_grad_mask = True
            self.evaluate_attn_mask = True
            self.evaluate_input_mask = True
        elif self.mask_type == "attention":
            self.evaluate_grad_mask = True
            self.evaluate_attn_mask = True
            self.evaluate_attn_signature_mask = True
        elif self.mask_type == "sample":
            self.evaluate_sample_cmi_mask = True
        else:
            raise NotImplementedError

    def eval_prediction(self, obs, action, next_obs):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs must be 1-dimensional
        :param action: (bs, action_dim) if self.continuous_state else (bs,)
        :param next_obs: {obs_i_key: (bs, obs_i_shape)}
        """
        obs, action, next_obs = self.preprocess(obs, action, next_obs)

        self.setup_mask_eval()

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_feature = self.encoder(next_obs)

        with torch.no_grad():
            if self.evaluate_mixture_cmi_mask:
                self.get_mixture_cmi_mask(obs, action, next_feature)

        with torch.set_grad_enabled(self.evaluate_grad_mask):
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

    def setup_mask_eval_for_reward(self):
        if self.intrinsic_reward_type in ["local_causality_uncertainty_binary",
                                          "local_causality_uncertainty_continuous", "cai"]:
            if self.local_causality_metric == "grad":
                self.evaluate_grad_mask = True
            elif self.local_causality_metric == "attention":
                self.evaluate_attn_mask = True
            elif self.local_causality_metric == "input_mask":
                assert self.mask_type == "input"
                self.evaluate_input_mask = True
            elif self.local_causality_metric == "attention_signature":
                assert self.mask_type == "attention"
                self.evaluate_attn_signature_mask = True
            elif self.local_causality_metric == "sample_cmi":
                assert self.mask_type == "sample"
                self.evaluate_sample_cmi_mask = True
            else:
                raise NotImplementedError

    def compute_prediction_uncertainty(self, pred_next_feature):
        # pred_next_feature: (bs, ensemble_size, feature_dim)
        if self.continuous_state:
            uncertainty = pred_next_feature.var(dim=-2)                                 # (bs, feature_dim)
        else:
            pred_next_feature = F.one_hot(pred_next_feature, self.input_dim)            # (bs, ensemble_size, feature_dim, input_dim)
            pred_next_feature = pred_next_feature.sum(dim=-3)                           # (bs, feature_dim, input_dim)
            prob = pred_next_feature / pred_next_feature.sum(dim=-1, keepdim=True)      # (bs, feature_dim, input_dim)
            logp = torch.log(prob.clamp(min=1e-20))                                     # (bs, feature_dim, input_dim)
            uncertainty = -(prob * logp).sum(dim=-1)                                    # (bs, feature_dim)
        return uncertainty

    def intrinsic_reward(self, pred_next_feature, next_feature):
        """
        :param pred_next_feature: (bs, ensemble_size, feature_dim)
        :param next_feature: (bs, feature_dim)
        :return reward: (bs,)
        """
        if self.intrinsic_reward_type == "dynamics_curiosity":
            # self.priority: (bs, ensemble_size, feature_dim)
            assert self.mask_type != "sample"
            reward = self.priority.mean(dim=(-2, -1))
        elif self.intrinsic_reward_type == "dynamics_uncertainty":
            uncertainty = self.compute_prediction_uncertainty(pred_next_feature)
            reward = uncertainty.mean(dim=-1)
        elif self.intrinsic_reward_type in ["local_causality_uncertainty_binary",
                                            "local_causality_uncertainty_continuous"]:
            for mask in [self.grad_mask, self.attn_mask, self.input_mask, self.attn_signature_mask, self.sample_cmi_mask]:
                if mask is not None:
                    break
            assert mask is not None

            # mask: (bs, ensemble_size, num_t_variables, num_tp1_variables)
            if self.intrinsic_reward_type == "local_causality_uncertainty_binary":
                mask = (mask >= self.local_causality_threshold).float()                 # (bs, ensemble_size, feature_dim, feature_dim + action_dim)

            mask_var = mask.var(dim=-3)                                                 # (bs, feature_dim, feature_dim + action_dim)

            if self.use_dynamics_uncertainty_mask:
                uncertainty = self.compute_prediction_uncertainty(pred_next_feature)    # (bs, ensemble_size, feature_dim)
                if self.new_uncertainty_calculation:
                    du_mask = uncertainty <= self.dynamics_uncertainty_mask_threshold       # (bs, feature_dim)
                    correct_mask = (pred_next_feature == next_feature.unsqueeze(dim=-2)).all(dim=-2)
                    mask_var[du_mask & correct_mask] = 0
                else:
                    du_mask = uncertainty > self.dynamics_uncertainty_mask_threshold        # (bs, feature_dim)
                    mask_var[~du_mask] = 0

            # idx_dict, begin = {}, 0
            # for k in self.dynamics_keys:
            #     dim = len(self.params.obs_dims[k])
            #     for i in range(dim):
            #         idx_dict[begin + i] = [k, i]
            #     begin += dim
            # idx_dict[begin] = "action"

            # for i in range(mask_var.shape[-2]):
            #     for j in range(mask_var.shape[-1]):
            #         if mask_var[0, i, j] > 0:
            #             print(idx_dict[i], idx_dict[j], mask_var[0, i, j], mask[0, :, i, j])

            reward = mask_var.mean(dim=(-2, -1))
        elif self.intrinsic_reward_type == "cai":
            assert self.sample_cmi_mask is not None

            # self.sample_cmi_mask: (bs, ensemble_size, feature_dim, num_objs + 1)
            reward = self.sample_cmi_mask[:, :, :, -1].mean(dim=(-2, -1))
        else:
            raise NotImplementedError

        return reward * self.intrinsic_reward_scale

    def eval_intrinsic_reward(self, obs, action, next_obs, done=None, info=None):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
            notice that bs must be 1-dimensional
        :param action: (bs, action_dim) if self.continuous_state else (bs,)
        :param next_obs: {obs_i_key: (bs, obs_i_shape)}
        """
        if self.intrinsic_reward_type == "none":
            return 0

        assert self.intrinsic_reward_type in ["dynamics_curiosity", "dynamics_uncertainty",
                                              "local_causality_uncertainty_binary", "cai",
                                              "local_causality_uncertainty_continuous"]

        return_np = isinstance(action, np.ndarray)
        if return_np:
            assert done is not None and info is not None
            obs, action, next_obs = self.preprocess(obs, action, next_obs, done, info)

        self.setup_mask_eval_for_reward()

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_feature = self.encoder(next_obs)

        with torch.set_grad_enabled(self.evaluate_grad_mask):
            # local causality is cached as self.xxx_mask during predict_step_with_feature
            pred_next_feature = self.predict_step_with_feature(obj_feature, feature, action, next_feature)

        # print("next_obs")
        # begin = 0
        # for k, v in next_obs.items():
        #     label = to_numpy(v[0])
        #     pred = to_numpy(pred_next_feature[0, :, begin:begin + len(v[0])])
        #     if (label != pred).any():
        #         print(k, label)
        #         print(pred)
        #     begin += len(label)
        # print()

        if not self.continuous_state:
            next_feature = [next_feature_i.argmax(dim=-1) for next_feature_i in next_feature]   # [(bs,)] * feature_dim
            next_feature = torch.stack(next_feature, dim=-1)                                    # (bs, feature_dim)

        reward = self.intrinsic_reward(pred_next_feature, next_feature)
        if return_np:
            reward = to_numpy(reward)

        self.reset_mask_eval()

        return reward

    def preprocess(self, obs, action, next_obs, done=None, info=None):
        key = self.dynamics_keys[0]
        if self.continuous_action and action.dtype != np.float32:
            action = action.astype(np.float32)
        if not self.continuous_action:
            if action.dtype != np.int64:
                action = action.astype(np.int64)
            assert action.ndim == obs[key].ndim - 1
            action = action[..., None]

        if done is not None and done.any():
            assert done.all()
            next_obs = {k: np.array([info_i["obs"][k] for info_i in info])
                        for k in self.dynamics_keys}

        action = torch.torch.tensor(action, device=self.device)
        obs = preprocess_obs(obs, self.params, key_type="dynamics_keys")
        obs = {k: torch.tensor(v, device=self.device) for k, v in obs.items()}
        next_obs = preprocess_obs(next_obs, self.params, key_type="dynamics_keys")
        next_obs = {k: torch.tensor(v, device=self.device) for k, v in next_obs.items()}

        return obs, action, next_obs

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
            print("EnsembleDynamicsGradAttn loaded from", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.global_causality_val = checkpoint["global_causality_val"]
            self.global_causality_mask = self.global_causality_val >= self.global_causality_threshold
