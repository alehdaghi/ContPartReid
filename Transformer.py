import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from model import weights_init_classifier, weights_init_kaiming


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, token_size , num_classes, dim, depth=6, heads=1, mlp_dim=1024, dim_head = 1024):
        super().__init__()

        # self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(token_size, dim)))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.bottleneck = nn.BatchNorm1d(token_size * dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.linear_head = nn.Linear(token_size * dim, num_classes, bias=False)
        self.linear_head.apply(weights_init_classifier)

    def forward(self, feats):


        # x = torch.stack(feats_arr, dim=0)
        x = feats + self.pos_embedding

        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)

        x = self.bottleneck(x)
        return x, self.linear_head(x)
