import torch
from torch._C import float32
import torch.nn as nn
from copy import deepcopy
import math



def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # residual
        return self.norm(self.fn(x, **kwargs) + x)


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# Attention
class Attention(nn.Module):
    def __init__(self, 
                 dim,
                 heads=8, 
                 dim_head = 32,
                 dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        # This judgment condition is multi-head or single-head
        # If it is multi-head, then project_out = True, 
        # that is, put the q output from multiple heads together and use FC to process again,
        # else project_out = False，No need to concatenate q, and no need for FC processing
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5     # 1 / sqrt(d_k)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # W_q, W_k, W_v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, N = x.shape[:2]
        # Input：x -> [B, N, C_in]
        # self.to_qkv gets the three variables of Q, K, V at a time, 
        # and then uses chunk to split into three points, each of which is [B, N, h*d]
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # [B, N, h*d] -> [B, N, h, d] -> [B, h, N, d]
        q = q.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Q*K^T / sqrt(d_k) : [B, h, N, d] X [B, h, d, N] = [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax
        attn = self.attend(dots)
        # softmax(Q*K^T / sqrt(d_k)) * V ：[B, h, N, N] X [B, h, N, d] = [B, h, N, d]
        out = torch.matmul(attn, v)
        # [B, h, N, d] -> [B, N, h*d]=[B, N, C_out], C_out = h*d
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        
        return self.to_out(out)


# Transformer
class Transformer(nn.Module):
    def __init__(self, 
                 dim,            # 输入X的特征长度
                 depth,          # Encoder的层数
                 heads,          # multi-head的个数
                 dim_head,       # 每个head的dim
                 mlp_dim=2048,   # FFN中的dim
                 dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # 注意，PreNorm里内部已经做了residual。
            x = attn(x) 
            x = ff(x)
        return x


class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
