import torch
import torch.nn as nn
from backbone import *
import math

class DeTR(nn.Module):
    def __init__(self, 
                 img_size=640,
                 num_classes=80,
                 trainable=False,
                 num_objects=100,
                 backbone='r50'):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.num_objects = num_objects

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

        
    # Position Embedding
    def position_embedding(self, h, w, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
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

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
    
