import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

import PIL
from pycocotools import mask as coco_mask



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


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 data_dir,
                 img_folder='train2017', 
                 ann_file='annotations/' +' instances_train2017.json',
                 transform=None,
                 return_masks=False):
        img_folder = os.path.join(data_dir, img_folder)
        ann_file = os.path.join(data_dir, ann_file)
        
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.transform = transform
        self.return_masks = return_masks


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        img, target = super(CocoDetection, self).__getitem__(index)
        image_id = self.ids[index]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.load_img_targets(img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks


    def load_img_targets(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target



if __name__ == "__main__":
    from transforms import TrainTransform, TestTransform
    import cv2

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # dataset
    dataset = CocoDetection(
                data_dir=coco_root,                
                transform=TrainTransform)
    
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
