"""
This file should implement model that takes as input current state and output the causal graph of the current state
The model can either be based on uncertainty or curiosity
This file also supports RND due to very similar structure
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.normal import Normal

from model.gumbel import gumbel_sigmoid
from model.modules import ChannelMHAttention
from model.dynamics_utils import reset_layer, forward_network, get_train_mask, get_eval_mask, expand_helper, \
    flatten_helper, reset_layer_orth

from utils.utils import to_numpy, preprocess_obs, weight_init


class EnsembleCausalPredictor(nn.Module):
    def __init__(self, encoder, obj_encoder, params):
        super(EnsembleCausalPredictor, self).__init__()

        self.encoder = encoder
        self.obj_encoder = obj_encoder

        self.params = params
        self.dynamics_keys = params.dynamics_keys
        self.device = device = params.device
        self.dynamics_params = dynamics_params = params.mask_gt_params

        self.continuous_state = params.continuous_state
        self.continuous_action = params.continuous_action
        self.use_prioritized_buffer = params.training_params.replay_buffer_params.prioritized_buffer
        self.orthogonal_initialization = dynamics_params["orthogonal_init"]

        self.num_pred_steps = dynamics_params.num_pred_steps
        if self.num_pred_steps > 1:
            raise NotImplementedError
        self.ensemble_size = dynamics_params.ensemble_size
        self.monolithic_arch = dynamics_params.monolithic_arch
        self.intrinsic_reward_type = getattr(dynamics_params, "intrinsic_reward_type", "none")
        self.intrinsic_reward_scale = getattr(dynamics_params, "intrinsic_reward_scale", 0)
        self.local_causality_metric = getattr(dynamics_params, "local_causality_metric", "grad")
        self.local_causality_threshold = getattr(dynamics_params, "local_causality_threshold", 0)
        self.use_average_intrinsic = dynamics_params.average_intrinsic

        if self.intrinsic_reward_type == "gt_causal_curiosity":
            assert self.ensemble_size == 1
        if "uncertainty" in self.intrinsic_reward_type:
            assert self.ensemble_size > 1
        if self.intrinsic_reward_type == "rnd":
            assert self.ensemble_size == 1
            self.rnd_params = self.dynamics_params.rnd_params

        self.grad_attn_params = dynamics_params.grad_attn_params

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
        self.feature_dim = params.feature_dim

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

        # for rnd, we directly start from raw state features
        if self.intrinsic_reward_type == "rnd":
            input_size = sum(self.obj_dim.values()) 
            hidden_dim = self.rnd_params.hidden_dim
            rnd_rep_dim = self.rnd_params.rnd_rep_dim
            self.predictor = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, rnd_rep_dim))
            self.target = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, rnd_rep_dim))
            for param in self.target.parameters():
                param.requires_grad = False
            self.target.apply(weight_init)
            self.predictor.apply(weight_init)


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
                self.signature_queries = nn.Parameter(
                    torch.FloatTensor(*channel_shape, num_queries, in_dim).uniform_(-b, b))

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

        # predictor
        in_dim = attn_out_dim
        for out_dim in grad_attn_params.predictor_fc_dims:
            self.predictor_weights.append(nn.Parameter(torch.zeros(*channel_shape, in_dim, out_dim)))
            self.predictor_biases.append(nn.Parameter(torch.zeros(*channel_shape, 1, out_dim)))
            in_dim = out_dim

        self.feature_inner_dim = obj_encoder.feature_inner_dim

        # We just need a single number to go through sigmoid
        final_size = self.feature_dim + 1
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

            input_i = input_i.permute(3, 1, 2, 0)  # (input_i_dim, ensemble_size, feature_dim, bs)
            inputs.append(input_i)

        x = pad_sequence(inputs)  # (input_dim, num_objs + 1, ensemble_size, feature_dim, bs)
        x = x.permute(2, 3, 1, 4, 0)  # (ensemble_size, feature_dim, num_objs + 1, bs, input_dim)

        sa_feature = forward_network(x, self.sa_feature_weights, self.sa_feature_biases)
        sa_feature = F.relu(sa_feature)

        return sa_feature  # (ensemble_size, feature_dim, num_objs + 1, bs, feature_embed_dim)

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
                attn_signatures_q = attn_signatures.permute(0, 1, 3, 2, 4).reshape(ensemble_size * feature_dim * bs,
                                                                                   num_objs_p_1, -1)
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
                    attn_score = attn_score.mean(dim=3).reshape(bs * ensemble_size * feature_dim, num_objs + 1,
                                                                num_objs + 1)
                    if attn_mask is None:
                        attn_mask = attn_score
                    else:
                        attn_mask = torch.bmm(attn_score, attn_mask)

                if self.evaluate_attn_signature_mask:
                    if attn_signature_mask is None:
                        attn_signature_mask = torch.exp(log_attn_mask)
                        attn_signature_mask = attn_signature_mask.reshape(ensemble_size * feature_dim * bs,
                                                                          num_objs + 1, num_objs + 1)
                    else:
                        attn_signature_mask = torch.bmm(attn_signature_mask, attn_signature_mask)

            if self.evaluate_attn_mask:
                attn_mask = attn_mask.reshape(bs, ensemble_size, feature_dim, num_objs + 1, num_objs + 1)
                self.attn_mask = attn_mask[:, :, np.arange(self.feature_dim), self.query_index]

            if self.evaluate_attn_signature_mask:
                attn_signature_mask = attn_signature_mask.reshape(ensemble_size, feature_dim, bs, num_objs + 1,
                                                                  num_objs + 1)
                attn_signature_mask = attn_signature_mask.permute(2, 0, 1, 3, 4)
                self.attn_signature_mask = attn_signature_mask[:, :, np.arange(self.feature_dim), self.query_index]

        # (ensemble_size, feature_dim, bs, out_dim)
        sa_feature = sa_feature[:, np.arange(feature_dim), self.query_index]

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

    def rnd_forward_step(self, state_feature):
        if self.continuous_state:
            pass
        else:
            state_feature = [sf.squeeze() for sf in state_feature]
            state_feature = torch.cat(state_feature, dim=-1)
        prediction_feature = self.predictor(state_feature)
        with torch.no_grad():
            target_feature = self.target(state_feature)

        return (prediction_feature, target_feature)

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
        if not self.continuous_action:
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

        if self.intrinsic_reward_type == "gt_causal_curiosity":
            bs, obj_feature, feature, action, _, _ = \
                self.preprocess_inputs(obj_feature, feature, action, next_feature, delta_features)

            flatten_bs = len(bs) > 1
            if flatten_bs:
                assert not self.training
                assert self.sampled_sa_mask is None or self.sampled_sa_mask.ndim == 4

            logits = self.forward_step(obj_feature, action)  # (bs, ensemble_size, featuredim, featuredim + 1)
            if flatten_bs:
                logits = logits.reshape(*bs, *logits.shape[-3:])

            prediction = logits

        elif self.intrinsic_reward_type == "rnd":
            # rnd only takes in the state features
            prediction = self.rnd_forward_step(feature)
        else:
            raise NotImplementedError

        return prediction

    def update(self, obs, actions, causal_gts, eval=False):
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
        :param gt
        :return: {"loss_name": loss_value}
        """
        bs, num_pred_steps = actions.shape[0], actions.shape[-2]

        if num_pred_steps > 1:
            raise NotImplementedError

        action = actions[..., 0, :]

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)

        prediction = self.forward(obj_feature, feature, action)

        pred_loss, priority, _ = \
            self.compute_pred_loss_and_causal_target(causal_gts, prediction)

        loss_detail = {"pred_loss": pred_loss}
        loss = pred_loss

        self.backprop(loss, loss_detail)

        if self.use_prioritized_buffer:
            raise NotImplementedError
            if self.expand_feature_dim:
                priority = priority.mean(dim=-1)

            if self.expand_ensemble_dim:
                priority = priority.mean(dim=-1 if self.expand_feature_dim else -2)

            loss_detail["priority"] = priority

        return loss_detail

    def compute_pred_loss_and_causal_target(self, causal_gt, prediction):

        assert self.ensemble_size == 1

        if self.intrinsic_reward_type == "rnd":
            predicted, target = prediction
            prediction_error = torch.square(target.detach() - predicted).mean(
                dim=-1, keepdim=True)
            return prediction_error.mean(), prediction_error.flatten(), None
        else:
            assert self.intrinsic_reward_type == "gt_causal_curiosity"

        # For causal gt that is directly passed in from info, convert it to tensor first
        if isinstance(causal_gt, list):
            causal_gt_label = torch.tensor(np.array(causal_gt), device=self.device, dtype=torch.float32)
            causal_gt_label = causal_gt_label.unsqueeze(1)  # (bs, ensemble_size, feature_dim, feature_dim + 1)
        else:
            causal_gt = causal_gt.squeeze(1)  # Not sure why, but need this reshape
            causal_gt_label = causal_gt

        logits = prediction  # (bs, ensemble_size, feature_dim, final_dim)
        sigmoid_pred = torch.sigmoid(logits)  # (bs, ensemble_size, feature_dim, final_dim)

        ce = F.binary_cross_entropy(sigmoid_pred, causal_gt_label, reduction='none')
        pred_loss = ce.mean()

        pred_causal = sigmoid_pred > 0.5  # (bs, ensemble_size, feature_dim)
        # This is actually a bit different from priority: this is just for intrinsic reward
        if self.use_average_intrinsic:
            pred_loss_by_env = ce.mean(dim=[1, 2, 3])
            priority = 1 - torch.exp(-pred_loss_by_env)
        else:
            pred_loss_by_env = ce.amax(dim=[1, 2, 3])
            priority = 1 - torch.exp(-pred_loss_by_env)

        return pred_loss, priority, pred_causal

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

    def predict_step_with_feature(self, obj_feature, feature, action, causal_gt):
        """
        :param feature: (bs, feature_dim)
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: pred_next_feature: (bs, feature_dim)
        """
        self.use_label_for_mask_eval = True
        if not self.use_label_for_mask_eval:
            raise NotImplementedError

        prediction = self.forward(obj_feature, feature, action)

        _, self.priority, pred_causal = \
            self.compute_pred_loss_and_causal_target(causal_gt, prediction)

        return pred_causal

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
            self.evaluate_mixture_cmi_mask = True
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
        # Evaluation is disabled for now
        raise NotImplementedError
        obs, action = self.preprocess(obs, action)

        self.setup_mask_eval()

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)
        next_feature = self.encoder(next_obs)


        pred_next_feature = self.predict_step_with_feature(obj_feature, feature, action, next_feature)
        pred_next_feature = pred_next_feature.detach()

        if not self.continuous_state:
            feature = [feature_i.argmax(dim=-1) for feature_i in feature]  # [(bs,)] * feature_dim
            feature = torch.stack(feature, dim=-1)  # (bs, feature_dim)

            next_feature = [next_feature_i.argmax(dim=-1) for next_feature_i in next_feature]  # [(bs,)] * feature_dim
            next_feature = torch.stack(next_feature, dim=-1)  # (bs, feature_dim)

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

    def intrinsic_reward(self, pred_causal):
        """
        :param pred_next_feature: (bs, ensemble_size, feature_dim)
        :return reward: (bs,)
        """
        if self.intrinsic_reward_type == "gt_causal_curiosity":
            reward = self.priority
        elif self.intrinsic_reward_type == "rnd":
            reward = self.priority
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

        assert self.intrinsic_reward_type in ["gt_causal_curiosity", "rnd"]

        return_np = isinstance(action, np.ndarray)
        if return_np:
            assert done is not None and info is not None
            obs, action = self.preprocess(obs, action)

        feature = self.encoder(obs)
        obj_feature = self.obj_encoder(obs)

        causal_gt = [i.get("local_causality", None) for i in info]
        pred_causal = self.predict_step_with_feature(obj_feature, feature, action, causal_gt)

        reward = self.intrinsic_reward(pred_causal)
        if return_np:
            reward = to_numpy(reward)

        self.reset_mask_eval()

        return reward

    def preprocess(self, obs, action):
        key = self.dynamics_keys[0]
        if self.continuous_action and action.dtype != np.float32:
            action = action.astype(np.float32)
        if not self.continuous_action:
            if action.dtype != np.int64:
                action = action.astype(np.int64)
            assert action.ndim == obs[key].ndim - 1
            action = action[..., None]

        action = torch.torch.tensor(action, device=self.device)
        obs = preprocess_obs(obs, self.params, key_type="dynamics_keys")
        obs = {k: torch.tensor(v, device=self.device) for k, v in obs.items()}

        return obs, action

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
            print("EnsembleCausalPredictor loaded from", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.global_causality_val = checkpoint["global_causality_val"]
            self.global_causality_mask = self.global_causality_val >= self.global_causality_threshold
