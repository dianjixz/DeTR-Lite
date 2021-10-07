import torch
import numpy as np
import torch.nn as nn
from .backbone import resnet18, resnet34, resnet50, resnet101
from .transformer import Transformer
from utils.modules import MLP
import utils.box_ops as box_ops
import math

class DeTR(nn.Module):
    def __init__(self, 
                 device,
                 img_size=640,
                 num_classes=80,
                 trainable=False,
                 conf_thresh=0.01,
                 nms_thresh=0.6,
                 num_heads=8,
                 num_encoders=6,
                 num_decoders=6,
                 num_queries=100,
                 hidden_dim=256,
                 mlp_dim=2048,
                 dropout=0.1,
                 aux_loss=False,
                 criterion=None,
                 backbone='r50',
                 use_nms=False):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.use_nms = use_nms


        # position embedding
        self.pos_embed = self.position_embedding(num_pos_feats=hidden_dim//2,
                                                 normalize=True)
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
    def position_embedding(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        h, w = self.img_size // 32, self.img_size // 32
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        # generate xy coord mat
        # y_embed = [[0, 0, 0, ...], [1, 1, 1, ...]...]
        # x_embed = [[0, 1, 2, ...], [0, 1, 2, ...]...]
        y_embed, x_embed = torch.meshgrid([torch.arange(1, h+1, dtype=torch.float32), 
                                        torch.arange(1, w+1, dtype=torch.float32)])
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (h + eps) * scale
            x_embed = x_embed / (w + eps) * scale
    
        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(self.device)
        x_embed = x_embed[None, :, :].to(self.device)


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


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


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
            outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)
        
        # train
        if self.trainable:
            # The loss is computed in the external file
            return outputs

        # test
        else:
            with torch.no_grad():
                # batch_size = 1
                out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
                # [B, N, C] -> [N, C]
                prob = out_logits[0].softmax(-1)
                scores, labels = prob[..., 1:].max(-1)

                # convert to [x0, y0, x1, y1] format
                bboxes = box_ops.box_cxcywh_to_xyxy(out_bbox)[0]
                bboxes = bboxes * self.img_size

                # intermediate outputs
                if 'aux_outputs' in outputs:
                    for i, aux_outputs in enumerate(outputs['aux_outputs']):
                        # batch_size = 1
                        out_logits_i, out_bbox_i = aux_outputs['pred_logits'], aux_outputs['pred_boxes']
                        # [B, N, C] -> [N, C]
                        prob_i = out_logits_i[0].softmax(-1)
                        scores_i, labels_i = prob_i[..., 1:].max(-1)

                        # convert to [x0, y0, x1, y1] format
                        bboxes_i = box_ops.box_cxcywh_to_xyxy(out_bbox_i)[0]
                        bboxes_i = bboxes_i * self.img_size

                        scores = torch.cat([scores, scores_i], dim=0)
                        labels = torch.cat([labels, labels_i], dim=0)
                        bboxes = torch.cat([bboxes, bboxes_i], dim=0)
                
                # to cpu
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                bboxes = bboxes.cpu().numpy()

                # threshold
                keep = np.where(scores >= self.conf_thresh)

                # nms
                if self.use_nms:
                    # nms
                    keep = np.zeros(len(bboxes), dtype=np.int)
                    for i in range(self.num_classes):
                        inds = np.where(labels == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = bboxes[inds]
                        c_scores = scores[inds]
                        c_keep = self.nms(c_bboxes, c_scores)
                        keep[inds[c_keep]] = 1

                    keep = np.where(keep > 0)
                    scores = scores[keep]
                    labels = labels[keep]
                    bboxes = bboxes[keep]

                
                return scores, labels, bboxes
