"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. infer on the origin dataset
3. infer on the augmentation dataset
4. save splited mining result with `torch.save(results, f'/out/mining_results_{RANK}.pt')`
5. merge mining result
"""
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process

from utils.general import scale_coords
from ymir.mining.util import (YmirDataset, collate_fn_with_fake_ann, load_image_file, load_image_file_with_ann,
                              update_consistency)
from ymir.ymir_yolov5 import YmirYolov5

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    gpu = LOCAL_RANK if LOCAL_RANK >= 0 else 0
    device = torch.device('cuda', gpu)
    ymir_yolov5.to(device)

    load_fn = partial(load_image_file, img_size=ymir_yolov5.img_size, stride=ymir_yolov5.stride)
    batch_size_per_gpu: int = ymir_yolov5.batch_size_per_gpu
    gpu_count: int = ymir_yolov5.gpu_count
    cpu_count: int = os.cpu_count() or 1
    num_workers_per_gpu = min([
        cpu_count // max(gpu_count, 1), batch_size_per_gpu if batch_size_per_gpu > 1 else 0,
        ymir_yolov5.num_workers_per_gpu
    ])

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    max_barrier_times = (len(images) // max(1, WORLD_SIZE)) // batch_size_per_gpu
    # origin dataset
    if RANK != -1:
        images_rank = images[RANK::WORLD_SIZE]
    else:
        images_rank = images
    origin_dataset = YmirDataset(images_rank, load_fn=load_fn)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=num_workers_per_gpu,
                                          pin_memory=ymir_yolov5.pin_memory,
                                          drop_last=False)

    results = []
    mining_results = dict()
    beta = 1.3
    dataset_size = len(images_rank)
    pbar = tqdm(origin_dataset_loader) if RANK == 0 else origin_dataset_loader
    for idx, batch in enumerate(pbar):
        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        with torch.no_grad():
            pred = ymir_yolov5.forward(batch['image'].float().to(device), nms=True)

        if RANK in [-1, 0]:
            write_ymir_monitor_process(ymir_cfg,
                                       task='mining',
                                       naive_stage_percent=0.3 * idx * batch_size_per_gpu / dataset_size,
                                       stage=YmirStage.TASK)
        preprocess_image_shape = batch['image'].shape[2:]
        for inner_idx, det in enumerate(pred):  # per image
            result_per_image = []
            image_file = batch['image_file'][inner_idx]
            if len(det):
                origin_image_shape = (batch['origin_shape'][0][inner_idx], batch['origin_shape'][1][inner_idx])
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                result_per_image.append(det)
            else:
                mining_results[image_file] = -beta
                continue

            results_per_image = torch.cat(result_per_image, dim=0).data.cpu().numpy()
            results.append(dict(image_file=image_file, origin_shape=origin_image_shape, results=results_per_image))

    aug_load_fn = partial(load_image_file_with_ann, img_size=ymir_yolov5.img_size, stride=ymir_yolov5.stride)
    aug_dataset = YmirDataset(results, load_fn=aug_load_fn)
    aug_dataset_loader = td.DataLoader(aug_dataset,
                                       batch_size=batch_size_per_gpu,
                                       shuffle=False,
                                       sampler=None,
                                       collate_fn=collate_fn_with_fake_ann,
                                       num_workers=num_workers_per_gpu,
                                       pin_memory=ymir_yolov5.pin_memory,
                                       drop_last=False)

    dataset_size = len(results)
    monitor_gap = max(1, dataset_size // 1000 // batch_size_per_gpu)
    pbar = tqdm(aug_dataset_loader) if RANK == 0 else aug_dataset_loader
    for idx, batch in enumerate(pbar):
        if idx % monitor_gap == 0 and RANK in [-1, 0]:
            write_ymir_monitor_process(ymir_cfg,
                                       task='mining',
                                       naive_stage_percent=0.3 + 0.7 * idx * batch_size_per_gpu / dataset_size,
                                       stage=YmirStage.TASK)

        batch_consistency = [0.0 for _ in range(len(batch['image_file']))]
        aug_keys = ['flip', 'cutout', 'rotate', 'resize']

        pred_result = dict()
        for key in aug_keys:
            with torch.no_grad():
                pred_result[key] = ymir_yolov5.forward(batch[f'image_{key}'].float().to(device), nms=True)

        for inner_idx in range(len(batch['image_file'])):
            for key in aug_keys:
                preprocess_image_shape = batch[f'image_{key}'].shape[2:]
                result_per_image = []
                det = pred_result[key][inner_idx]
                if len(det) == 0:
                    # no result for the image with augmentation f'{key}'
                    batch_consistency[inner_idx] += beta
                    continue

                # prediction result from origin image
                fake_ann = batch['results_list'][inner_idx]
                # bboxes = fake_ann[:, :4].data.cpu().numpy().astype(np.int32)
                conf = fake_ann[:, 4]

                # augmentated bbox from bboxes, aug_conf = conf
                aug_bboxes_key = batch[f'bboxes_{key}_list'][inner_idx].astype(np.int32)

                origin_image_shape = (batch[f'origin_shape_{key}'][0][inner_idx],
                                      batch[f'origin_shape_{key}'][1][inner_idx])

                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(preprocess_image_shape, det[:, :4], origin_image_shape).round()
                result_per_image.append(det)

                pred_bboxes_key = det[:, :4].data.cpu().numpy().astype(np.int32)
                pred_conf_key = det[:, 4].data.cpu().numpy()
                batch_consistency[inner_idx] = update_consistency(consistency=batch_consistency[inner_idx],
                                                                  consistency_per_aug=2.0,
                                                                  beta=beta,
                                                                  pred_bboxes_key=pred_bboxes_key,
                                                                  pred_conf_key=pred_conf_key,
                                                                  aug_bboxes_key=aug_bboxes_key,
                                                                  aug_conf=conf)

        for inner_idx in range(len(batch['image_file'])):
            batch_consistency[inner_idx] /= len(aug_keys)
            image_file = batch['image_file'][inner_idx]
            mining_results[image_file] = batch_consistency[inner_idx]

    torch.save(mining_results, f'/out/mining_results_{max(0,RANK)}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    ymir_yolov5 = YmirYolov5(ymir_cfg)

    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the mining result
    if WORLD_SIZE > 1:
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
