"""
data augmentations for CALD method, including horizontal_flip, rotate(5'), cutout
official code: https://github.com/we1pingyu/CALD/blob/master/cald/cald_helper.py
"""
import os
import random
import sys
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import init_dist
from mmdet.apis.test import collect_results_gpu
from mmdet.utils.util_ymir import BBOX, CV_IMAGE
from nptyping import NDArray
from scipy.stats import entropy
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process
from ymir_infer import YmirModel

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def intersect(boxes1: BBOX, boxes2: BBOX) -> NDArray:
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    max_xy = np.minimum(
        np.expand_dims(boxes1[:, 2:], axis=1).repeat(n2, axis=1),
        np.expand_dims(boxes2[:, 2:], axis=0).repeat(n1, axis=0))

    min_xy = np.maximum(
        np.expand_dims(boxes1[:, :2], axis=1).repeat(n2, axis=1),
        np.expand_dims(boxes2[:, :2], axis=0).repeat(n1, axis=0))
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=None)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def horizontal_flip(image: CV_IMAGE, bbox: BBOX) \
        -> Tuple[CV_IMAGE, BBOX]:
    """
    image: opencv image, [height,width,channels]
    bbox: numpy.ndarray, [N,4] --> [x1,y1,x2,y2]
    """
    image = image.copy()

    width = image.shape[1]
    # Flip image horizontally
    image = image[:, ::-1, :]
    if len(bbox) > 0:
        bbox = bbox.copy()
        # Flip bbox horizontally
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
    return image, bbox


def cutout(image: CV_IMAGE,
           bbox: BBOX,
           cut_num: int = 2,
           fill_val: int = 0,
           bbox_remove_thres: float = 0.4,
           bbox_min_thres: float = 0.1) -> Tuple[CV_IMAGE, BBOX]:
    '''
        Cutout augmentation
        image: A PIL image
        boxes: bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        fill_val: Value filled in cut out
        bbox_remove_thres: Theshold to remove bbox cut by cutout

        Out: new image, new_boxes, new_labels
    '''
    image = image.copy()
    bbox = bbox.copy()

    if len(bbox) == 0:
        return image, bbox

    original_h, original_w, original_channel = image.shape
    count = 0
    for _ in range(50):
        # Random cutout size: [0.15, 0.5] of original dimension
        cutout_size_h = random.uniform(0.05 * original_h, 0.2 * original_h)
        cutout_size_w = random.uniform(0.05 * original_w, 0.2 * original_w)

        # Random position for cutout
        left = random.uniform(0, original_w - cutout_size_w)
        right = left + cutout_size_w
        top = random.uniform(0, original_h - cutout_size_h)
        bottom = top + cutout_size_h
        cutout = np.array([[float(left), float(top), float(right), float(bottom)]])

        # Calculate intersect between cutout and bounding boxes
        overlap_size = intersect(cutout, bbox)
        area_boxes = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        ratio = overlap_size / (area_boxes + 1e-14)
        # If all boxes have Iou greater than bbox_remove_thres, try again
        if ratio.max() > bbox_remove_thres or ratio.max() < bbox_min_thres:
            continue

        image[int(top):int(bottom), int(left):int(right), :] = fill_val
        count += 1
        if count >= cut_num:
            break
    return image, bbox


def rotate(image: CV_IMAGE, bbox: BBOX, rot: float = 5) -> Tuple[CV_IMAGE, BBOX]:
    image = image.copy()
    bbox = bbox.copy()
    h, w, c = image.shape
    center = np.array([w / 2.0, h / 2.0])
    s = max(h, w) * 1.0
    trans = get_affine_transform(center, s, rot, [w, h])
    if len(bbox) > 0:
        for i in range(bbox.shape[0]):
            x1, y1 = affine_transform(bbox[i, :2], trans)
            x2, y2 = affine_transform(bbox[i, 2:], trans)
            x3, y3 = affine_transform(bbox[i, [2, 1]], trans)
            x4, y4 = affine_transform(bbox[i, [0, 3]], trans)
            bbox[i, :2] = [min(x1, x2, x3, x4), min(y1, y2, y3, y4)]
            bbox[i, 2:] = [max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
    image = cv2.warpAffine(image, trans, (w, h), flags=cv2.INTER_LINEAR)
    return image, bbox


def get_3rd_point(a: NDArray, b: NDArray) -> NDArray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point: NDArray, rot_rad: float) -> List:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def transform_preds(coords: NDArray, center: NDArray, scale: Any, rot: float, output_size: List) -> NDArray:
    trans = get_affine_transform(center, scale, rot, output_size, inv=True)
    target_coords = affine_transform(coords, trans)
    return target_coords


def get_affine_transform(center: NDArray,
                         scale: Any,
                         rot: float,
                         output_size: List,
                         shift: NDArray = np.array([0, 0], dtype=np.float32),
                         inv: bool = False) -> NDArray:
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir(np.array([0, src_w * -0.5], np.float32), rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt: NDArray, t: NDArray) -> NDArray:
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def resize(img: CV_IMAGE, boxes: BBOX, ratio: float = 0.8) -> Tuple[CV_IMAGE, BBOX]:
    """
    ratio: <= 1.0
    """
    assert ratio <= 1.0, f'resize ratio {ratio} must <= 1.0'

    h, w, _ = img.shape
    ow = int(w * ratio)
    oh = int(h * ratio)
    resize_img = cv2.resize(img, (ow, oh))
    new_img = np.zeros_like(img)
    new_img[:oh, :ow] = resize_img

    if len(boxes) == 0:
        return new_img, boxes
    else:
        return new_img, boxes * ratio


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


def split_result(result: NDArray) -> Tuple[BBOX, NDArray, NDArray]:
    if len(result) > 0:
        bboxes = result[:, :4].astype(np.int32)
        conf = result[:, 4]
        class_id = result[:, 5]
    else:
        bboxes = np.zeros(shape=(0, 4), dtype=np.int32)
        conf = np.zeros(shape=(0, 1), dtype=np.float32)
        class_id = np.zeros(shape=(0, 1), dtype=np.int32)

    return bboxes, conf, class_id


class CALDMiner(YmirModel):
    def mining(self):
        with open(self.cfg.ymir.input.candidate_index_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]

        max_barrier_times = len(images) // WORLD_SIZE
        if RANK == -1:
            N = len(images)
            tbar = tqdm(images)
        else:
            images_rank = images[RANK::WORLD_SIZE]
            N = len(images_rank)
            if RANK == 0:
                tbar = tqdm(images_rank)
            else:
                tbar = images_rank

        monitor_gap = max(1, N // 100)
        idx = -1
        beta = 1.3
        mining_result = []
        for idx, asset_path in enumerate(tbar):
            if idx % monitor_gap == 0 and RANK in [0, -1]:
                write_ymir_monitor_process(self.cfg, task='mining', naive_stage_percent=idx / N, stage=YmirStage.TASK)

            # batch-level sync, avoid 30min time-out error
            if WORLD_SIZE > 1 and idx < max_barrier_times:
                dist.barrier()

            img = cv2.imread(asset_path)
            # xyxy,conf,cls
            result = self.predict(img)
            bboxes, conf, _ = split_result(result)
            if len(result) == 0:
                # no result for the image without augmentation
                mining_result.append((asset_path, -beta))
                continue

            consistency = 0.0
            aug_bboxes_dict, aug_results_dict = self.aug_predict(img, bboxes)
            for key in aug_results_dict:
                # no result for the image with augmentation f'{key}'
                if len(aug_results_dict[key]) == 0:
                    consistency += beta
                    continue

                bboxes_key, conf_key, _ = split_result(aug_results_dict[key])
                cls_scores_aug = 1 - conf_key
                cls_scores = 1 - conf

                consistency_per_aug = 2.0
                ious = get_ious(bboxes_key, aug_bboxes_dict[key])
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
                    consistency_cls = 0.5 * \
                        (conf[origin_idx] + conf_key[aug_idx]) * (1 - js)
                    consistency_per_inst = abs(consistency_box + consistency_cls - beta)
                    consistency_per_aug = min(consistency_per_aug, consistency_per_inst.item())

                    consistency += consistency_per_aug

            consistency /= len(aug_results_dict)

            mining_result.append((asset_path, consistency))

        if WORLD_SIZE > 1:
            mining_result = collect_results_gpu(mining_result, len(images))

        return mining_result

    def predict(self, img: CV_IMAGE) -> NDArray:
        """
        predict single image and return bbox information
        img: opencv BGR, uint8 format
        """
        results = self.infer(img)

        xyxy_conf_idx_list = []
        for idx, result in enumerate(results):
            for line in result:
                if any(np.isinf(line)):
                    continue
                x1, y1, x2, y2, score = line
                xyxy_conf_idx_list.append([x1, y1, x2, y2, score, idx])

        if len(xyxy_conf_idx_list) == 0:
            return np.zeros(shape=(0, 6), dtype=np.float32)
        else:
            return np.array(xyxy_conf_idx_list, dtype=np.float32)

    def aug_predict(self, image: CV_IMAGE, bboxes: BBOX) -> Tuple[Dict[str, BBOX], Dict[str, NDArray]]:
        """
        for different augmentation methods: flip, cutout, rotate and resize
            augment the image and bbox and use model to predict them.

        return the predict result and augment bbox.
        """
        aug_dict: Dict[str, Callable] = dict(flip=horizontal_flip, cutout=cutout, rotate=rotate, resize=resize)

        aug_bboxes = dict()
        aug_results = dict()
        for key in aug_dict:
            aug_img, aug_bbox = aug_dict[key](image, bboxes)

            aug_result = self.predict(aug_img)
            aug_bboxes[key] = aug_bbox
            aug_results[key] = aug_result

        return aug_bboxes, aug_results


def main():
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    cfg = get_merged_config()
    miner = CALDMiner(cfg)
    gpu = max(0, LOCAL_RANK)
    device = torch.device('cuda', gpu)
    miner.model.to(device)
    mining_result = miner.mining()

    if RANK in [0, -1]:
        rw.write_mining_result(mining_result=mining_result)
        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1, stage=YmirStage.POSTPROCESS)

    return 0


if __name__ == "__main__":
    sys.exit(main())
