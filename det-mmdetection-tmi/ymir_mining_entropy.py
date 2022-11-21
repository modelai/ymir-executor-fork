"""
entropy mining
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import init_dist
from mmdet.apis.test import collect_results_gpu
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process
from ymir_mining_cald import split_result, CALDMiner

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class EntropyMiner(CALDMiner):

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
                mining_result.append((asset_path, -10))
                continue
            conf = conf.data.cpu().numpy()
            mining_result.append((asset_path, -np.sum(conf * np.log2(conf))))

        if WORLD_SIZE > 1:
            mining_result = collect_results_gpu(mining_result, len(images))

        return mining_result


def main():
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    cfg = get_merged_config()
    miner = EntropyMiner(cfg)
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
