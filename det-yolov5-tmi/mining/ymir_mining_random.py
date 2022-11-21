"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. infer on the origin dataset
3. infer on the augmentation dataset
4. save splited mining result with `torch.save(results, f'/out/mining_results_{RANK}.pt')`
5. merge mining result
"""
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config

from mining.util import (YmirDataset, collate_fn_with_fake_ann, load_image_file, load_image_file_with_ann,
                         update_consistency)
from utils.general import scale_coords
from utils.ymir_yolov5 import YmirYolov5

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    gpu = LOCAL_RANK if LOCAL_RANK >= 0 else 0
    device = torch.device('cuda', gpu)
    ymir_yolov5.to(device)
    
    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    images_rank = images[RANK::WORLD_SIZE]
    mining_results=dict()
    for image in images_rank:
        mining_results[image] = random.random()

    torch.save(mining_results, f'/out/mining_results_{RANK}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    ymir_yolov5 = YmirYolov5(ymir_cfg, task='mining')

    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the mining result
    if LOCAL_RANK != -1:
        dist.barrier()

    if RANK in [0, -1]:
        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/mining_results_{rank}.pt'))

        ymir_mining_result = []
        for result in results:
            for img_file, score in result.items():
                ymir_mining_result.append((img_file, score))
        rw.write_mining_result(mining_result=ymir_mining_result)

    if LOCAL_RANK != -1:
        print(f'rank: {RANK}, start destroy process group')
        # dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
