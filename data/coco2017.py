import os
import numpy as np
import torch
from torch.utils.data import Dataset

import PIL
from pycocotools.coco import COCO


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_root = '/mnt/share/ssd2/dataset/COCO/'


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 data_dir='COCO', 
                 transform=None, 
                 json_file='instances_train2017.json',
                 name='train2017'):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        # augmentation
        self.transform = transform


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt


    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img =PIL.Image.open(img_file)  
        # cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img =PIL.Image.open(img_file)  
            # cv2.imread(img_file)

        return img, id_


    def pull_anno(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        
        target = []
        for anno in annotations:
            if 'bbox' in anno:
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = xmin + anno['bbox'][2]
                ymax = ymin + anno['bbox'][3]
                
                if anno['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')
        return target


    def load_img_targets(self, index):
        anno_ids = self.coco.getAnnIds(imgIds=[int(index)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(index) + '.jpg')
        img =PIL.Image.open(img_file)  
        # cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(index) + '.jpg')
            img =PIL.Image.open(img_file)  
            # cv2.imread(img_file)

        assert img is not None

        height, width = img.size
        
        # COCOAnnotation Transform
        # start here :
        labels = []
        boxes = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:   
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = np.min((width - 1, xmin + np.max((0, anno['bbox'][2] - 1))))
                ymax = np.min((height - 1, ymin + np.max((0, anno['bbox'][3] - 1))))

                if xmax > xmin and ymax > ymin:
                    # a valid target
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)
                    labels.append(cls_id)
                    boxes.append([xmin, ymin, xmax, ymax])


            else:
                print('No bbox !!!')
        # end here .
        targets = {"labels": labels,
                   "boxes":  torch.from_numpy(boxes).float()}

        return img, targets, height, width


    def pull_item(self, index):
        # load a image
        id_ = self.ids[index]
        img, targets, height, width = self.load_img_targets(id_)

        # check targets
        if targets is None:
            targets["labels": 80] # the background index
            targets["boxes": [0, 0, 0, 0]]

        # augment
        img, targets = self.transform(img, targets)

        return img, targets, height, width


if __name__ == "__main__":
    from transforms import TrainTransform, TestTransform
    import cv2

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # dataset
    dataset = COCODataset(
                data_dir=coco_root,
                transform=TrainTransform,
                debug=False)
    
    for i in range(1000):
        img, targets, h, w = dataset.pull_item(i)
        # [C, H, W] -> [H, W, C], tensor -> numpy
        img = img.permute(1, 2, 0).numpy()
        img = (img.numpy() * std + mean) * 255. 
        img = img[:, :, (2, 1, 0)].astype(np.uint8)
        img_h, img_w = img.shape[:2]

        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')
        cls_gt = targets["labels"]
        box_gt = targets["boxes"]
        for i in range(len(cls_gt)):
            cls_id = cls_gt[i]
            x1, y1, x2, y2 = box_gt[i]
            x1 = x1 * img_w
            y1 = y1 * img_h
            x2 = x2 * img_w
            y2 = y2 * img_h
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cls_id = coco_class_index[int(cls_id)]
            cls_name = coco_class_labels[cls_id]
            mess = '%s' % (cls_name)
            # mess = '%s' % (cls_name)
            cv2.putText(img, mess, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('gt', img)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
