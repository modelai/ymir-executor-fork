"""
utils function for ymir and yolov5
"""
import json
import os
import os.path as osp
import random
import shutil
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from nptyping import NDArray, Shape, UInt8
from ymir_exc import result_writer as rw
from ymir_exc.util import get_bool, get_weight_files

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    weight_files = get_weight_files(cfg, suffix=('.pt'))
    # choose weight file by priority, best.pt > xxx.pt
    for p in weight_files:
        if p.endswith('best.pt'):
            return p

    if len(weight_files) > 0:
        return max(weight_files, key=osp.getctime)

    return ""


class YmirYolov5(torch.nn.Module):
    """
    used for mining and inference to init detector and predict.
    """

    def __init__(self, cfg: edict):
        super().__init__()
        self.cfg = cfg

        self.gpu_id: str = str(cfg.param.get('gpu_id', '0'))
        device = select_device(self.gpu_id)  # will set CUDA_VISIBLE_DEVICES=self.gpu_id
        self.gpu_count: int = len(self.gpu_id.split(',')) if self.gpu_id else 0
        self.batch_size_per_gpu: int = int(cfg.param.get('batch_size_per_gpu', 4))
        self.num_workers_per_gpu: int = int(cfg.param.get('num_workers_per_gpu', 4))
        self.pin_memory: bool = get_bool(cfg, 'pin_memory', False)
        self.batch_size: int = self.batch_size_per_gpu * self.gpu_count
        self.model = self.init_detector(device)
        self.model.eval()
        self.device = device
        self.class_names: List[str] = cfg.param.class_names
        self.stride = self.model.stride
        self.conf_thres: float = float(cfg.param.conf_thres)
        self.iou_thres: float = float(cfg.param.iou_thres)

        img_size = int(cfg.param.img_size)
        imgsz = [img_size, img_size]
        imgsz = check_img_size(imgsz, s=self.stride)

        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
        self.img_size: List[int] = imgsz

    def extract_feats(self, x):
        """
        return the feature maps before sigmoid for mining
        """
        return self.model.model(x)[1]

    def forward(self, x, nms=False):
        pred = self.model(x)
        if not nms:
            return pred

        pred = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=None,  # not filter class_idx
            agnostic=False,
            max_det=100)
        return pred

    def init_detector(self, device: torch.device) -> DetectMultiBackend:
        weights = get_weight_file(self.cfg)

        if not weights:
            raise Exception("no weights file specified!")

        data_yaml = osp.join(self.cfg.ymir.output.root_dir, 'data.yaml')
        model = DetectMultiBackend(
            weights=weights,
            device=device,
            dnn=False,  # not use opencv dnn for onnx inference
            data=data_yaml)  # dataset.yaml path

        return model

    def predict(self, img: CV_IMAGE) -> NDArray:
        """
        predict single image and return bbox information
        img: opencv BGR, uint8 format
        """
        # preprocess: padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # preprocess: convert data format
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)

        img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
        img1.unsqueeze_(dim=0)  # expand for batch dim
        pred = self.forward(img1, nms=True)

        result = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
                result.append(det)

        # xyxy, conf, cls
        if len(result) > 0:
            tensor_result = torch.cat(result, dim=0)
            numpy_result = tensor_result.data.cpu().numpy()
        else:
            numpy_result = np.zeros(shape=(0, 6), dtype=np.float32)

        return numpy_result

    def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
        anns = []
        result = self.predict(img)

        for i in range(result.shape[0]):
            xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
            ann = rw.Annotation(class_name=self.class_names[int(cls)],
                                score=conf,
                                box=rw.Box(x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

            anns.append(ann)

        return anns


def convert_ymir_to_yolov5(cfg: edict, out_dir: str = None):
    """
    convert ymir format dataset to yolov5 format
    generate data.yaml for training/mining/infer
    """

    out_dir = out_dir or cfg.ymir.output.root_dir
    data = dict(path=out_dir, nc=len(cfg.param.class_names), names=cfg.param.class_names)
    for split, prefix in zip(['train', 'val', 'test'], ['training', 'val', 'candidate']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        if osp.exists(src_file):
            shutil.copy(src_file, f'{out_dir}/{split}.tsv')

        data[split] = f'{split}.tsv'

    with open(osp.join(out_dir, 'data.yaml'), 'w') as fw:
        fw.write(yaml.safe_dump(data))


def get_attachments(cfg: edict) -> Dict[str, List[str]]:
    # 4. add attachments for quantification
    attachments: Dict[str, List[str]] = dict()
    attachments['images'] = []
    attachments['configs'] = []
    with open(cfg.ymir.input.val_index_file, 'r') as fp:
        img_files = [line.split()[0] for line in fp.readlines()]

    random.seed(25)
    random.shuffle(img_files)
    models_dir: str = cfg.ymir.output.models_dir
    img_size: int = int(cfg.param.img_size)
    attachments_image_dir = osp.join(models_dir, 'attachments/images')
    os.makedirs(attachments_image_dir, exist_ok=True)
    for img_f in img_files[0:200]:
        shutil.copy(img_f, attachments_image_dir)
        attachments['images'].append(osp.basename(img_f))

    attachments_config_dir = osp.join(models_dir, 'attachments/configs')
    os.makedirs(attachments_config_dir, exist_ok=True)
    for config_f in ['preconfig.json', 'postconfig.json']:
        with open(config_f, 'r') as fp:
            quant_config = json.load(fp)

        if config_f == 'preconfig.json':
            quant_config['inputs'][0]['dims'] = [1, 3, img_size, img_size]
        else:
            from models.experimental import attempt_load  # scoped to avoid circular import
            quant_model = attempt_load(f'{models_dir}/best.pt', map_location='cpu')
            anchors = quant_model.model[-1].anchors * quant_model.stride.view(-1, 1, 1)
            np_anchors = anchors.data.cpu().numpy().reshape(3, -1).astype(int)
            quant_config['anchor0'] = np_anchors[0].tolist()
            quant_config['anchor0'] = np_anchors[1].tolist()
            quant_config['anchor0'] = np_anchors[2].tolist()

        out_config_f = osp.join(attachments_config_dir, config_f)
        with open(out_config_f, 'w') as fw:
            json.dump(quant_config, fw)

        # save to yaml with relative path to mdoels_dir
        attachments['configs'].append(osp.basename(config_f))

    return attachments
