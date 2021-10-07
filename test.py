import numpy as np
import cv2
import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn

from data import *
from data import config

from models.detr import DeTR

parser = argparse.ArgumentParser(description='DeTR Detection')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--use_nms', action='store_true', default=False, 
                    help='use nms.')
parser.add_argument('--no_aux_loss', action='store_true',
                    help="Disables auxiliary decoding losses (loss at each layer)")
parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')

# model 
parser.add_argument('-bk', '--backbone', default='r50', type=str, 
                    help='backbone')
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--mlp_dim', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s: %.3f' % (class_names[int(cls_indx)], scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                mess = '%s: %.3f' % (cls_name, scores[i])
                # mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset='voc'):
    num_images = len(testset)
    save_path = os.path.join('det_results/images/', args.dataset)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img_raw, _ = testset.pull_image(index)
        h, w, _ = img_raw.shape
        size = np.array([[w, h, w, h]])

        # preprocess
        img, _, _, scale, offset = transform(img_raw)
        x = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1).float()
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # map the boxes to original image
        bboxes -= offset
        bboxes /= scale
        bboxes *= size

        img_processed = vis(img_raw, 
                            bboxes, 
                            scores, 
                            cls_inds, 
                            thresh, 
                            class_colors, 
                            class_names, 
                            class_indexs, 
                            dataset)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)
        print('Saving the' + str(index) + '-th image ...')
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
    input_size = args.img_size

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=VOC_ROOT, 
                               img_size=args.img_size,
                               image_sets=[('2007', 'test')])

    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(data_dir=coco_root,
                              img_size=args.img_size,
                              json_file='instances_val2017.json',
                              name='val2017')

    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    # model
    model_name = args.version
    print('Model: ', model_name)

    # config file
    cfg = config.detr

    # build model
    model = DeTR(device=device,
                 batch_size=1,
                 img_size=args.img_size,
                 num_classes=num_classes,
                 trainable=False,
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 num_heads=args.nheads,
                 num_encoders=args.enc_layers,
                 num_decoders=args.dec_layers,
                 num_queries=args.num_queries,
                 hidden_dim=args.hidden_dim,
                 mlp_dim=args.mlp_dim,
                 dropout=0.,
                 aux_loss=not args.no_aux_loss,
                 backbone=args.backbone,
                 use_nms=args.use_nms).to(device)


    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    model.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=model, 
        device=device, 
        testset=dataset,
        transform=BaseTransform(input_size),
        thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset=args.dataset
        )
