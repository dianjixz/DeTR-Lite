import torch
import torch.nn as nn
from data import *
import argparse
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from evaluator.cocoapi_evaluator import COCOAPIEvaluator

from models.detr import DeTR


parser = argparse.ArgumentParser(description='Detection Transformer')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r34, r50, r101')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('-size', '--input_size', default=640, type=int,
                    help='input_size')
parser.add_argument('--trained_model', type=str,
                    default='weights/', 
                    help='Trained state_dict file path to open')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')

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




def voc_test(model, device, input_size):
    evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                device=device,
                                transform=BaseTransform(input_size),
                                labelmap=VOC_CLASSES,
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, input_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size)
                        )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # build model
    model = DeTR(device=device,
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

    # load net
    model.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    model.eval()
    print('Finished loading model!')
    model = model.to(device)
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, device, input_size)
        elif args.dataset == 'coco-val':
            coco_test(model, device, input_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, device, input_size, test=True)
