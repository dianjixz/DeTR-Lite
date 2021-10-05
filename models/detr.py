import torch
import torch.nn as nn
from torch.nn.modules import dropout
from backbone import *
from utils.modules import Transformer, MLP
import math

class DeTR(nn.Module):
    def __init__(self, 
                 device,
                 batch_size=1,
                 img_size=640,
                 num_classes=80,
                 trainable=False,
                 num_heads=8,
                 num_encoders=6,
                 num_decoders=6,
                 num_queries=100,
                 hidden_dim=256,
                 mlp_dim=2048,
                 aux_loss=False,
                 backbone='r50'):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss

        # position embedding
        self.pos_embed = self.position_embedding(batch_size, 
                                                 self.img_size // 32,
                                                 self.img_size // 32,
                                                 num_pos_feats=hidden_dim//2)
        # object query
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # backbone
        if backbone == 'r18':
            self.backbone = resnet18(pretrained=trainable, freeze_bn=trainable)
            c5 = 512
        elif backbone == 'r34':
            self.backbone = resnet34(pretrained=trainable, freeze_bn=trainable)
            c5 = 512
        elif backbone == 'r50':
            self.backbone = resnet50(pretrained=trainable, freeze_bn=trainable)
            c5 = 2048
        elif backbone == 'r101':
            self.backbone = resnet101(pretrained=trainable, freeze_bn=trainable)
            c5 = 2048
        
        # to compress channel of C5
        self.input_proj = nn.Conv2d(c5, hidden_dim, kernel_size=1)

        
        # transformer
        self.transformer = Transformer(dim=hidden_dim,
                                       num_encoders=num_encoders,
                                       num_decoders=num_decoders,
                                       num_heads=num_heads,
                                       dim_head=hidden_dim // num_heads,
                                       mlp_dim=mlp_dim,
                                       dropout=dropout,
                                       act='relu')

        # det
        self.cls_det = nn.Linear(hidden_dim, num_classes + 1)
        self.reg_det = MLP(hidden_dim, hidden_dim, 4, 3)

        
    # Position Embedding
    def position_embedding(self, B, H, W, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        h, w = self.img_size // 32, self.img_size // 32
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        # generate xy coord mat
        # y_embed = [[0, 0, 0, ...], [1, 1, 1, ...]...]
        # x_embed = [[0, 1, 2, ...], [0, 1, 2, ...]...]
        y_embed, x_embed = torch.meshgrid([torch.arange(h, dtype=torch.float32), 
                                        torch.arange(w, dtype=torch.float32)])
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
    
        # [H, W] -> [B, H, W]
        y_embed = y_embed.repeat(B, 1, 1)
        x_embed = x_embed.repeat(B, 1, 1)


        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        # torch.div(a, b, rounding_mode='floor') == (a // b)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [B, d, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos


    def forward(self, x):
        # backbone
        x = self.backbone(x)
        x = self.input_proj(x)

        # transformer
        tgt = torch.zeros_like(self.query_embed)
        h = self.transformer(x, tgt, self.pos_embed, self.query_embed)

        # det
        cls_det = self.cls_det(h)
        reg_det = self.reg_det(h).sigmoid()

        if self.trainable:
            # compute loss
            pass

        else:
            # decode cls and reg prediction
            pass
