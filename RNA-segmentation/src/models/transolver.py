import torch
import torch.nn as nn
from einops import rearrange

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, embed_dim, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError
        act = ACTIVATION[act]

        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, embed_dim), act())
        self.linear_post = nn.Linear(embed_dim, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(embed_dim, embed_dim), act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_Block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4, num_basis=32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.SelfAttn = Physics_Attention_Irregular_Mesh(
            embed_dim=hidden_dim, num_heads=num_heads, num_basis=num_basis, dropout=dropout,
        )
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, x, return_attention=False):
        _x, attn = self.SelfAttn(self.ln_1(x), return_attention=return_attention)
        x = _x + x
        x = self.mlp(self.ln_2(x)) + x
        return x, attn


class Physics_Attention(nn.Module):
    def __init__(self, in_dim=3, embed_dim=256, num_heads=8, n_layers=4,
                 dropout=0., num_basis=64, mlp_ratio=4, act='gelu'):
        super().__init__()
        self.__name__ = 'Physics_Attention'
        self.n_layers = n_layers

        self.preprocess = MLP(in_dim, embed_dim=embed_dim, n_output=embed_dim, n_layers=1)
        self.blocks = nn.ModuleList([
            Transolver_Block(
                hidden_dim=embed_dim, num_heads=num_heads, num_basis=num_basis,
                dropout=dropout, mlp_ratio=mlp_ratio, act=act,
            )
            for _ in range(n_layers)
        ])

        self.placeholder = nn.Parameter((1 / embed_dim) * torch.rand(embed_dim))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_attention=False):
        attentions = [] if return_attention else None

        x = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            x, C_yx = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(C_yx)

        return (x, attentions) if return_attention else x


class Transolver(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, n_layers=4, dropout=0.1, n_class=260,
                 num_basis=64, mlp_ratio=4, act='gelu', input_features='xyz',
                 last_activation=None):
        super().__init__()
        self.last_activation = last_activation

        C_in = {'xyz': 3, 'hks': 16}[input_features]
        self.feature_extractor = Physics_Attention(
            in_dim=C_in, embed_dim=embed_dim, num_heads=num_heads,
            n_layers=n_layers, dropout=dropout, num_basis=num_basis,
            mlp_ratio=mlp_ratio, act=act,
        )
        self.last_lin = nn.Linear(embed_dim, n_class)

    def forward(self, verts, mass, L=None, evals=None, evecs=None,
                gradX=None, gradY=None, faces=None, edges=None):
        if len(verts.shape) == 2:
            appended_batch_dim = True
            verts = verts.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L is not None: L = L.unsqueeze(0)
            if evals is not None: evals = evals.unsqueeze(0)
            if evecs is not None: evecs = evecs.unsqueeze(0)
            if gradX is not None: gradX = gradX.unsqueeze(0)
            if gradY is not None: gradY = gradY.unsqueeze(0)
            if edges is not None: edges = edges.unsqueeze(0)
            if faces is not None: faces = faces.unsqueeze(0)
        else:
            appended_batch_dim = False

        feat = self.feature_extractor(verts, return_attention=False)
        feat = self.last_lin(feat)

        if self.last_activation is not None:
            feat = self.last_activation(feat)
        if appended_batch_dim:
            feat = feat.squeeze(0)
        return feat