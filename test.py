import numpy as np
import cv2
import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn

from data import *
from data.transforms import ValTransforms

from models.detr import DeTR

parser = argparse.ArgumentParser(description='DeTR Detection')
# dataset
parser.add_argument('-root', '--data_root', default='/mnt/share/ssd2/dataset',
                    help='root to dataset')
parser.add_argument('-d', '--dataset', default='coco',
                    help='voc or coco')
# basic
parser.add_argument('-size', '--img_size', default=800, type=int,
                    help='img_size')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--show', action='store_true', default=False, 
                    help='show visualization results.')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
# model 
parser.add_argument('-bk', '--backbone', default='r50', type=str, 
                    help='backbone')
parser.add_argument('--num_encoders', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--num_decoders', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--mlp_dim', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--num_heads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--no_aux_loss', action='store_true',
                    help="Disables auxiliary decoding losses (loss at each layer)")
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--use_nms', action='store_true', default=False, 
                    help='use nms.')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(args,
         net, 
         device, 
         dataset,
         transforms=None,
         vis_thresh=0.4, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         show=False,
         dataset_name='coco'):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        h, w, _ = image.shape
        scale = np.array([[w, h, w, h]])

        # prepare
        x = transforms(image)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes *= scale

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=dataset_name
                            )
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    img_size = args.img_size

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(data_dir=data_dir, image_sets=[('2007', 'test')])

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.data_root, 'COCO')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    json_file='instances_val2017.json')
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = DeTR(args=args,
                 device=device,
                 img_size=args.img_size,
                 num_classes=num_classes,
                 trainable=False,
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 aux_loss=False,
                 backbone=args.backbone,
                 use_nms=args.use_nms).to(device)

    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    model.to(device).eval()
    print('Finished loading model!')

    # run
    test(args=args,
        net=model, 
        device=device, 
        dataset=dataset,
        transforms=ValTransforms(args.img_size),
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        dataset_name=args.dataset)
