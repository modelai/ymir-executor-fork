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
from easydict import EasyDict as edict
from mmcv.runner import init_dist
from mmdet.apis.test import collect_results_gpu
from mmdet.utils.util_ymir import BBOX, CV_IMAGE
from nptyping import NDArray
from scipy.stats import entropy
from tqdm import tqdm
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, get_ymir_process
from ymir_infer import YmirModel

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))



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


class YmirMining(YmirModel):

    def __init__(self, cfg: edict):
        super().__init__(cfg)
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            mining_task_idx = 0
            # infer_task_idx = 1
            task_num = 2
        else:
            mining_task_idx = 0
            # infer_task_idx = 0
            task_num = 1
        self.task_idx = mining_task_idx
        self.task_num = task_num

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

        mining_result = []
        for idx, asset_path in enumerate(tbar):
            if idx % monitor_gap == 0:
                percent = get_ymir_process(stage=YmirStage.TASK,
                                           p=idx / N,
                                           task_idx=self.task_idx,
                                           task_num=self.task_num)
                monitor.write_monitor_logger(percent=percent)
            # batch-level sync, avoid 30min time-out error
            if WORLD_SIZE > 1 and idx < max_barrier_times:
                dist.barrier()

            img = cv2.imread(asset_path)
            # xyxy,conf,cls
            result = self.predict(img)
            bboxes, conf, _ = split_result(result)
            if len(result) == 0:
                # no result for the image without augmentation
                mining_result.append((asset_path, -10))
                continue
            conf = conf.data.cpu().numpy()
            mining_result.append((asset_path, -np.sum(conf * np.log2(conf))))

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



def main():
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    cfg = get_merged_config()
    miner = YmirMining(cfg)
    gpu = max(0, LOCAL_RANK)
    device = torch.device('cuda', gpu)
    miner.model.to(device)
    mining_result = miner.mining()

    if RANK in [0, -1]:
        rw.write_mining_result(mining_result=mining_result)

        percent = get_ymir_process(stage=YmirStage.POSTPROCESS, p=1, task_idx=miner.task_idx, task_num=miner.task_num)
        monitor.write_monitor_logger(percent=percent)

    return 0


if __name__ == "__main__":
    sys.exit(main())
