import os
import random
import sys

import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from mmcv.runner import init_dist
from mmdet.apis.test import collect_results_gpu
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class RandomMiner(object):

    def __init__(self, cfg: edict):
        if LOCAL_RANK != -1:
            init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

        self.cfg = cfg
        gpu = max(0, LOCAL_RANK)
        self.device = f'cuda:{gpu}'

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
            if idx % monitor_gap == 0:
                write_ymir_monitor_process(cfg=self.cfg,
                                           task='mining',
                                           naive_stage_percent=idx / N,
                                           stage=YmirStage.TASK,
                                           task_order='tmi')

            if WORLD_SIZE > 1 and idx < max_barrier_times:
                dist.barrier()

            with torch.no_grad():
                consistency = self.compute_score(asset_path=asset_path)
            mining_result.append((asset_path, consistency))

        if WORLD_SIZE > 1:
            mining_result = collect_results_gpu(mining_result, len(images))

        if RANK in [0, -1]:
            rw.write_mining_result(mining_result=mining_result)
            write_ymir_monitor_process(cfg=self.cfg,
                                       task='mining',
                                       naive_stage_percent=1,
                                       stage=YmirStage.POSTPROCESS,
                                       task_order='tmi')
        return mining_result

    def compute_score(self, asset_path: str) -> float:
        return random.random()


def main():
    cfg = get_merged_config()
    miner = RandomMiner(cfg)
    miner.mining()
    return 0


if __name__ == "__main__":
    sys.exit(main())
