"""run.py:
img --(model)--> pred --(augmentation)--> (aug1_pred, aug2_pred, ..., augN_pred)
img --(augmentation)--> aug1_img --(model)--> pred1
img --(augmentation)--> aug2_img --(model)--> pred2
...
img --(augmentation)--> augN_img --(model)--> predN

dataload(img) --(model)--> pred
dataload(img, pred) --(augmentation1)--> (aug1_img, aug1_pred) --(model)--> pred1

1. split dataset with DDP sampler
2. use DDP model to infer sampled dataloader
3. gather infer result

"""
import os
from typing import Any, List

import cv2
import numpy as np
import torch.utils.data as td
from nptyping import NDArray
from scipy.stats import entropy
from torch.utils.data._utils.collate import default_collate
from utils.augmentations import letterbox
from ymir.mining.data_augment import cutout, horizontal_flip, intersect, resize, rotate
from ymir.ymir_yolov5 import BBOX

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_ious(boxes1: BBOX, boxes2: BBOX) -> NDArray:
    """
    args:
        boxes1: np.array, (N, 4), xyxy
        boxes2: np.array, (M, 4), xyxy
    return:
        iou: np.array, (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    iner_area = intersect(boxes1, boxes2)
    area1 = area1.reshape(-1, 1).repeat(area2.shape[0], axis=1)
    area2 = area2.reshape(1, -1).repeat(area1.shape[0], axis=0)
    iou = iner_area / (area1 + area2 - iner_area + 1e-14)
    return iou


def preprocess(img, img_size, stride):
    img1 = letterbox(img, img_size, stride=stride, auto=False)[0]

    # preprocess: convert data format
    img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img1 = np.ascontiguousarray(img1)
    # img1 = torch.from_numpy(img1).to(self.device)

    img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
    return img1


def load_image_file(img_file: str, img_size, stride):
    img = cv2.imread(img_file)
    img1 = letterbox(img, img_size, stride=stride, auto=False)[0]

    # preprocess: convert data format
    img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img1 = np.ascontiguousarray(img1)
    # img1 = torch.from_numpy(img1).to(self.device)

    img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
    # img1.unsqueeze_(dim=0)  # expand for batch dim
    return dict(image=img1, origin_shape=img.shape[0:2], image_file=img_file)
    # return img1


def load_image_file_with_ann(image_info: dict, img_size, stride):
    img_file = image_info['image_file']
    # xyxy(int) conf(float) class_index(int)
    bboxes = image_info['results'][:, :4].astype(np.int32)
    img = cv2.imread(img_file)
    aug_dict = dict(flip=horizontal_flip, cutout=cutout, rotate=rotate, resize=resize)

    data = dict(image_file=img_file, origin_shape=img.shape[0:2])
    for key in aug_dict:
        aug_img, aug_bbox = aug_dict[key](img, bboxes)
        preprocess_aug_img = preprocess(aug_img, img_size, stride)
        data[f'image_{key}'] = preprocess_aug_img
        data[f'bboxes_{key}'] = aug_bbox
        data[f'origin_shape_{key}'] = aug_img.shape[0:2]

    data.update(image_info)
    return data


def collate_fn_with_fake_ann(batch):
    new_batch = dict()
    for key in ['flip', 'cutout', 'rotate', 'resize']:
        new_batch[f'bboxes_{key}_list'] = [data[f'bboxes_{key}'] for data in batch]

        new_batch[f'image_{key}'] = default_collate([data[f'image_{key}'] for data in batch])

        new_batch[f'origin_shape_{key}'] = default_collate([data[f'origin_shape_{key}'] for data in batch])

    new_batch['results_list'] = [data['results'] for data in batch]
    new_batch['image_file'] = [data['image_file'] for data in batch]

    return new_batch


def update_consistency(consistency, consistency_per_aug, beta, pred_bboxes_key, pred_conf_key, aug_bboxes_key,
                       aug_conf):
    cls_scores_aug = 1 - pred_conf_key
    cls_scores = 1 - aug_conf

    consistency_per_aug = 2.0
    ious = get_ious(pred_bboxes_key, aug_bboxes_key)
    aug_idxs = np.argmax(ious, axis=0)
    for origin_idx, aug_idx in enumerate(aug_idxs):
        max_iou = ious[aug_idx, origin_idx]
        if max_iou == 0:
            consistency_per_aug = min(consistency_per_aug, beta)
        p = cls_scores_aug[aug_idx]
        q = cls_scores[origin_idx]
        m = (p + q) / 2.
        js = 0.5 * entropy([p, 1 - p], [m, 1 - m]) + 0.5 * entropy([q, 1 - q], [m, 1 - m])
        if js < 0:
            js = 0
        consistency_box = max_iou
        consistency_cls = 0.5 * (aug_conf[origin_idx] + pred_conf_key[aug_idx]) * (1 - js)
        consistency_per_inst = abs(consistency_box + consistency_cls - beta)
        consistency_per_aug = min(consistency_per_aug, consistency_per_inst.item())

        consistency += consistency_per_aug
    return consistency


class YmirDataset(td.Dataset):
    def __init__(self, images: List[Any], load_fn=None):
        super().__init__()
        self.images = images
        self.load_fn = load_fn

    def __getitem__(self, index):
        return self.load_fn(self.images[index])

    def __len__(self):
        return len(self.images)
