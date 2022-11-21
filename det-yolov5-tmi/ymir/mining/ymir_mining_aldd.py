"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. infer on the origin dataset
3. infer on the augmentation dataset
4. save splited mining result with `torch.save(results, f'/out/mining_results_{RANK}.pt')`
5. merge mining result
"""
import os
import sys
import warnings
from functools import partial
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as td
from easydict import EasyDict as edict
from tqdm import tqdm
from ymir.mining.util import YmirDataset, load_image_file
from ymir.ymir_yolov5 import YmirYolov5
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class ALDD(object):

    def __init__(self, ymir_cfg: edict):
        self.avg_pool_size = 9
        self.max_pool_size = 32
        self.avg_pool_pad = (self.avg_pool_size - 1) // 2

        self.num_classes = len(ymir_cfg.param.class_names)
        if ymir_cfg.param.get('class_distribution_scores', ''):
            scores = [float(x.strip()) for x in ymir_cfg.param.class_distribution_scores.split(',')]
            if len(scores) < self.num_classes:
                warnings.warn('extend 1.0 to class_distribution_scores')
                scores.extend([1.0] * (self.num_classes - len(scores)))
            self.class_distribution_scores = np.array(scores[0:self.num_classes], dtype=np.float32)
        else:
            self.class_distribution_scores = np.array([1.0] * self.num_classes, dtype=np.float32)

    def calc_unc_val(self, heatmap: torch.Tensor) -> torch.Tensor:
        # mean of entropy
        ent = F.binary_cross_entropy(heatmap, heatmap, reduction='none')
        avg_ent = F.avg_pool2d(ent,
                               kernel_size=self.avg_pool_size,
                               stride=1,
                               padding=self.avg_pool_pad,
                               count_include_pad=False)  # N, 1, H, W
        mean_of_entropy = torch.sum(avg_ent, dim=1, keepdim=True)  # N, 1, H, W

        # entropy of mean
        avg_heatmap = F.avg_pool2d(heatmap,
                                   kernel_size=self.avg_pool_size,
                                   stride=1,
                                   padding=self.avg_pool_pad,
                                   count_include_pad=False)  # N, C, H, W
        ent_avg = F.binary_cross_entropy(avg_heatmap, avg_heatmap, reduction='none')
        entropy_of_mean = torch.sum(ent_avg, dim=1, keepdim=True)  # N, 1, H, W

        uncertainty = entropy_of_mean - mean_of_entropy
        unc = F.max_pool2d(uncertainty,
                           kernel_size=self.max_pool_size,
                           stride=self.max_pool_size,
                           padding=0,
                           ceil_mode=False)

        # aggregating
        scores = torch.mean(unc, dim=(1, 2, 3))  # (N,)
        return scores

    def compute_aldd_score(self, net_output: List[torch.Tensor], net_input_shape: Any):
        """
        args:
            imgs: list[np.array(H, W, C)]
        returns:
            scores: list of float
        """
        if not isinstance(net_input_shape, (list, tuple)):
            net_input_shape = (net_input_shape, net_input_shape)

        # CLASS_DISTRIBUTION_SCORE = np.array([1.0] * num_of_class)
        scores_list = []

        for feature_map in net_output:
            feature_map.sigmoid_()

        for each_class_index in range(self.num_classes):
            feature_map_list: List[torch.Tensor] = []

            # each_output_feature_map: [bs, 3, h, w, 5 + num_classes]
            for each_output_feature_map in net_output:
                net_output_conf = each_output_feature_map[:, :, :, :, 4]
                net_output_cls_mult_conf = net_output_conf * each_output_feature_map[:, :, :, :, 5 + each_class_index]
                # feature_map_reshape: [bs, 3, h, w]
                feature_map_reshape = F.interpolate(net_output_cls_mult_conf,
                                                    net_input_shape,
                                                    mode='bilinear',
                                                    align_corners=False)
                feature_map_list.append(feature_map_reshape)

            # len(net_output) = 3
            # feature_map_concate: [bs, 9, h, w]
            feature_map_concate = torch.cat(feature_map_list, 1)
            # scores: [bs, 1] for each class
            scores = self.calc_unc_val(feature_map_concate)
            scores = scores.cpu().detach().numpy()
            scores_list.append(scores)

        # total_scores: [bs, num_classes]
        total_scores = np.stack(scores_list, axis=1)
        total_scores = total_scores * self.class_distribution_scores
        total_scores = np.sum(total_scores, axis=1)

        return total_scores


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

    max_barrier_times = len(images) // WORLD_SIZE // batch_size_per_gpu
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

    mining_results = dict()
    dataset_size = len(images_rank)
    pbar = tqdm(origin_dataset_loader) if RANK == 0 else origin_dataset_loader
    miner = ALDD(ymir_cfg)
    for idx, batch in enumerate(pbar):
        # batch-level sync, avoid 30min time-out error
        if WORLD_SIZE > 1 and idx < max_barrier_times:
            dist.barrier()

        with torch.no_grad():
            featuremap_output = ymir_yolov5.model.model(batch['image'].float().to(device))[1]
            unc_scores = miner.compute_aldd_score(featuremap_output, ymir_yolov5.img_size)

        for each_imgname, each_score in zip(batch["image_file"], unc_scores):
            mining_results[each_imgname] = each_score

        if RANK in [-1, 0]:
            write_ymir_monitor_process(ymir_cfg,
                                       task='mining',
                                       naive_stage_percent=idx * batch_size_per_gpu / dataset_size,
                                       stage=YmirStage.TASK)

    torch.save(mining_results, f'/out/mining_results_{max(0,RANK)}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    # note select_device(gpu_id) will set os.environ['CUDA_VISIBLE_DEVICES'] to gpu_id
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
