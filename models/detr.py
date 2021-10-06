import torch
import torch.nn as nn
from torch.nn.modules import dropout
from .backbone import resnet18, resnet34, resnet50, resnet101
from .transformer import Transformer
from utils.modules import MLP
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
                 dropout=0.1,
                 aux_loss=False,
                 criterion=None,
                 backbone='r50'):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.criterion = criterion


        # position embedding
        self.pos_embed = self.position_embedding(batch_size, 
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
                                       dropout=self.dropout,
                                       act='relu',
                                       return_intermediate=True)

        # det
        self.cls_det = nn.Linear(hidden_dim, num_classes + 1)
        self.reg_det = MLP(hidden_dim, hidden_dim, 4, 3)

        
    # Position Embedding
    def position_embedding(self, B, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
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


    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def forward(self, x):
        # backbone
        x = self.backbone(x)
        x = self.input_proj(x)

        # transformer
        h = self.transformer(x, self.pos_embed, self.query_embed.weight)[0]

        # output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
        outputs_class = self.cls_det(h)
        outputs_coord = self.reg_det(h).sigmoid()

        # we only compute the loss of last output from decoder
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        if self.trainable:
            # The loss is computed in the external file
            return outputs

        else:
            # post process
            scores = None
            cls_inds = None
            bboxes = None

            return scores, cls_inds, bboxes
