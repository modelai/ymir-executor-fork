import argparse
import os
import os.path as osp
import sys
import warnings
from typing import Any, List

import cv2
import numpy as np
import torch.distributed as dist
from easydict import EasyDict as edict
from mmcv import DictAction
from mmcv.runner import init_dist
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_merged_config, write_ymir_monitor_process)

from mmdet.apis import inference_detector, init_detector
from mmdet.apis.test import collect_results_gpu
from mmdet.utils.util_ymir import get_best_weight_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_option(cfg_options: str) -> dict:
    parser = argparse.ArgumentParser(description='parse cfg options')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')

    args = parser.parse_args(f'--cfg-options {cfg_options}'.split())
    return args.cfg_options


def mmdet_result_to_ymir(results: List[Any], class_names: List[str]) -> List[rw.Annotation]:
    """
    results: List[NDArray[Shape['*,5'], Any]]
    """
    ann_list = []
    for idx, result in enumerate(results):
        for line in result:
            if any(np.isinf(line)):
                continue
            x1, y1, x2, y2, score = line
            ann = rw.Annotation(class_name=class_names[idx],
                                score=score,
                                box=rw.Box(x=round(x1), y=round(y1), w=round(x2 - x1), h=round(y2 - y1)))
            ann_list.append(ann)
    return ann_list


def get_config_file(cfg):
    if cfg.ymir.run_training:
        model_params_path: List = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path: List = cfg.param.get('model_params_path', [])

    model_dir = cfg.ymir.input.models_dir
    config_files = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(('.py'))
    ]

    if len(config_files) > 0:
        if len(config_files) > 1:
            warnings.warn(f'multiple config file found! use {config_files[0]}')
        return config_files[0]
    else:
        raise Exception(f'no config_file found in {model_dir} and {model_params_path}')


class YmirModel:
    def __init__(self, cfg: edict):
        self.cfg = cfg

        # Specify the path to model config and checkpoint file
        config_file = get_config_file(cfg)
        checkpoint_file = get_best_weight_file(cfg)
        options = cfg.param.get('cfg_options', None)
        cfg_options = parse_option(options) if options else None

        # current infer can only use one gpu!!!
        # gpu_ids = cfg.param.get('gpu_id', '0')
        # gpu_id = gpu_ids.split(',')[0]
        gpu_id = max(0, RANK)
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device=f'cuda:{gpu_id}', cfg_options=cfg_options)

    def infer(self, img):
        return inference_detector(self.model, img)


def main():
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    cfg = get_merged_config()

    with open(cfg.ymir.input.candidate_index_file, 'r') as f:
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
    infer_result = dict()
    model = YmirModel(cfg)

    # write infer result
    monitor_gap = max(1, N // 100)
    conf_threshold = float(cfg.param.conf_threshold)
    for idx, asset_path in enumerate(tbar):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        raw_anns = mmdet_result_to_ymir(result, cfg.param.class_names)

        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        infer_result[asset_path] = [ann for ann in raw_anns if ann.score >= conf_threshold]

        if idx % monitor_gap == 0:
            write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    if WORLD_SIZE > 1:
        infer_result = collect_results_gpu(infer_result, len(images))

    rw.write_infer_result(infer_result=infer_result)
    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
