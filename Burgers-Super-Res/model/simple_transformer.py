"""
Ported from galerkin-transformer/libs/model.py
"""
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from model.layers import default, Identity, FeedForward, SpectralConv1d, PositionalEncoding


class FunctionalMap_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, n_mode, n_dim, n_head, alpha_init=2.0):
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.dim_head = n_dim // n_head

        self.in_proj = nn.Linear(n_dim + 1, n_dim, bias=False)
        self.in_project_fx = nn.Linear(n_dim + 1, n_dim, bias=False)
        self.in_project_slice = nn.Linear(self.dim_head, n_mode, bias=False)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.temperature = nn.Parameter(torch.ones([1, n_head, 1, 1]) * 0.5)
        self.softmax = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.out_proj = nn.Linear(n_dim, n_dim, bias=False)

        self.alpha = nn.Parameter(torch.ones([1, n_head, 1, 1]) * alpha_init)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('I_d', torch.eye(self.dim_head).unsqueeze(0))

    def forward(self, x, xx, xxx, pos=None, weight=None):
        if weight is not None:
                x, xx = weight*x, weight*xx
        if pos is not None:
            x = torch.cat([x, pos], dim=-1)

        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.n_head, self.dim_head).permute(0, 2, 1, 3)
        x_mid  = self.in_proj(x) .reshape(B, N, self.n_head, self.dim_head).permute(0, 2, 1, 3)

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm  = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5).unsqueeze(-1)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        kH  = k.transpose(2, 3)
        kTk = torch.matmul(kH, k)
        alpha = self.sigmoid(self.alpha)
        reg = (1 - alpha) * kTk + alpha * self.I_d
        Z = torch.linalg.solve(reg, kH)
        C = torch.matmul(q, Z)

        out_token = torch.matmul(C, v)
        out = torch.einsum("bhgc,bhng->bhnc", out_token, slice_weights)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        return self.out_proj(out), C


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim, num_layers=2,
                 spacial_fc=False, spacial_dim=1, dropout=0.1,
                 activation='silu', return_latent=False, debug=False):
        super().__init__()
        dropout = default(dropout, 0.0)
        self.spacial_fc = spacial_fc
        activ = nn.SiLU() if activation == 'silu' else nn.ReLU()
        if spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), activ)])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(nn.Linear(n_hidden, n_hidden), activ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent

    def forward(self, x, grid=None):
        if self.spacial_fc:
            x = self.fc(torch.cat([x, grid], dim=-1))
        for layer in self.ff:
            x = self.dropout(layer(x))
        x = self.out(x)
        return x if not self.return_latent else (x, None)


class SpectralRegressor(nn.Module):
    def __init__(self, in_dim, n_hidden, freq_dim, out_dim, modes,
                 num_spectral_layers=2, n_grid=None, dim_feedforward=None,
                 spacial_fc=False, spacial_dim=1, return_freq=False,
                 return_latent=False, normalizer=None, activation='silu',
                 last_activation=True, dropout=0.1, debug=False):
        super().__init__()
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        if spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([
            SpectralConv1d(in_dim=n_hidden, out_dim=freq_dim, n_grid=n_grid,
                           modes=modes, dropout=dropout, activation=activation,
                           return_freq=return_freq, debug=debug)
        ])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(
                SpectralConv1d(in_dim=freq_dim, out_dim=freq_dim, n_grid=n_grid,
                               modes=modes, dropout=dropout, activation=activation,
                               return_freq=return_freq, debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()
        dim_feedforward = default(dim_feedforward, 2 * spacial_dim * freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent

    def forward(self, x, edge=None, pos=None, grid=None):
        x_latent, x_fts = [], []
        if self.spacial_fc:
            x = self.fc(torch.cat([x, grid], dim=-1))
        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)
            if self.return_latent:
                x_latent.append(x.contiguous())
        x = self.regressor(x)
        if self.normalizer:
            x = self.normalizer.inverse_transform(x)
        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        return x


ADDITIONAL_ATTR = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation',
                   'downscaler_activation', 'upscaler_activation',
                   'dropout', 'encoder_dropout', 'decoder_dropout', 'ffn_dropout',
                   'debug']


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=1,
                 dim_feedforward=512,
                 attention_type='functional',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init=1e-2,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False):
        super().__init__()
        dropout = default(dropout, 0.05)
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True

        self.attn = FunctionalMap_Attention_Irregular_Mesh(n_mode=64, n_dim=d_model, n_head=1)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        self.ff = FeedForward(in_dim=d_model, dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm, activation=activation_type,
                              dropout=ffn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type
        self.add_pos_emb = pos_emb
        if pos_emb:
            self.pos_emb = PositionalEncoding(d_model)
        self.attn_weight = attn_weight
        self.__name__ = 'FuncAttnIrregularEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        if self.add_pos_emb:
            x = self.pos_emb(x.permute(1, 0, 2)).permute(1, 0, 2)

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(x, x, x, pos=pos, weight=weight)
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x = x + self.dropout2(self.ff(x))
        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer'

    def forward(self, node, edge, pos, grid=None, weight=None):
        x_latent = []
        attn_weights = []

        x = self.feat_extract(node, edge)

        if self.spacial_residual or self.return_latent:
            res = x.contiguous()
            x_latent.append(res)

        for encoder in self.encoder_layers:
            if self.return_attn_weight:
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            else:
                x = encoder(x, pos, weight)
            if self.return_latent:
                x_latent.append(x.contiguous())

        if self.spacial_residual:
            x = res + x

        x_freq = self.freq_regressor(x)[:, :self.pred_len, :] if self.n_freq_targets > 0 else None
        x = self.dpo(x)
        x = self.regressor(x, grid=grid)

        return dict(preds=x, preds_freq=x_freq,
                    preds_latent=x_latent, attn_weights=attn_weights)

    def _initialize(self):
        self._get_feature()
        self._get_encoder()
        if self.n_freq_targets > 0:
            self._get_freq_regressor()
        self._get_regressor()
        if self.decoder_type in ['pointwise', 'convolution']:
            self._initialize_layer(self.regressor)
        self.config = dict(self.config)

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])
        self.dim_feedforward = default(self.dim_feedforward, 2 * self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.00)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'cosine', 'galerkin',
                                 'linear', 'softmax', 'functional']

    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)

    def _get_encoder(self):
        encoder_layer = SimpleTransformerEncoderLayer(
            d_model=self.n_hidden,
            n_head=self.n_head,
            attention_type=self.attention_type,
            dim_feedforward=self.dim_feedforward,
            layer_norm=self.layer_norm,
            attn_norm=self.attn_norm,
            norm_type=self.norm_type,
            batch_norm=self.batch_norm,
            pos_dim=self.pos_dim,
            xavier_init=self.xavier_init,
            diagonal_weight=self.diagonal_weight,
            symmetric_init=self.symmetric_init,
            attn_weight=self.return_attn_weight,
            residual_type=self.residual_type,
            activation_type=self.attn_activation,
            dropout=self.encoder_dropout,
            ffn_dropout=self.ffn_dropout,
            debug=self.debug,
        )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_freq_regressor(self):
        self.freq_regressor = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_freq_targets),
        )

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(
                in_dim=self.n_hidden, n_hidden=self.n_hidden,
                out_dim=self.n_targets, spacial_fc=self.spacial_fc,
                spacial_dim=self.spacial_dim, activation=self.regressor_activation,
                dropout=self.decoder_dropout, debug=self.debug)
        elif self.decoder_type == 'ifft':
            self.regressor = SpectralRegressor(
                in_dim=self.n_hidden, n_hidden=self.n_hidden,
                freq_dim=self.freq_dim, out_dim=self.n_targets,
                num_spectral_layers=self.num_regressor_layers,
                modes=self.fourier_modes, spacial_dim=self.spacial_dim,
                spacial_fc=self.spacial_fc, dim_feedforward=self.freq_dim,
                activation=self.regressor_activation, dropout=self.decoder_dropout)
        else:
            raise NotImplementedError(f"decoder_type '{self.decoder_type}' not implemented")

    def get_encoder(self):
        return self.encoder_layers

