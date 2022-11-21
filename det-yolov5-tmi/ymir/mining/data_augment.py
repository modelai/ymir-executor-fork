"""
data augmentations for CALD method, including horizontal_flip, rotate(5'), cutout
official code: https://github.com/we1pingyu/CALD/blob/master/cald/cald_helper.py
"""
import random
from typing import Any, List, Tuple

import cv2
import numpy as np
from nptyping import NDArray

from ymir.ymir_yolov5 import BBOX, CV_IMAGE


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
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
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
