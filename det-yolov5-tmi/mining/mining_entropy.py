"""
Consistency-based Active Learning for Object Detection CVPR 2022 workshop
official code: https://github.com/we1pingyu/CALD/blob/master/cald_train.py
"""
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from easydict import EasyDict as edict
from mining.data_augment import cutout, horizontal_flip, intersect, resize, rotate
from nptyping import NDArray
from scipy.stats import entropy
from tqdm import tqdm
from utils.ymir_yolov5 import BBOX, CV_IMAGE, YmirYolov5
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, get_ymir_process

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

class MiningEntropy(YmirYolov5):
    def __init__(self, cfg: edict):
        super().__init__(cfg)

        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            mining_task_idx = 0
            task_num = 2
        else:
            mining_task_idx = 0
            task_num = 1

        self.task_idx = mining_task_idx
        self.task_num = task_num

    def mining(self) -> List:
        N = dr.items_count(env.DatasetType.CANDIDATE)
        monitor_gap = max(1, N // 1000)
        idx = -1
        beta = 1.3
        mining_result = []
        for asset_path, _ in tqdm(dr.item_paths(dataset_type=env.DatasetType.CANDIDATE)):
            img = cv2.imread(asset_path)
            # xyxy,conf,cls
            result = self.predict(img,nms=False)
            bboxes, conf, _ = split_result(result)
            if len(result) == 0:
                # no result for the image without augmentation
                mining_result.append((asset_path, -10))
                continue
            mining_result.append((asset_path,-np.sum(conf*np.log2(conf))))
            idx += 1
            if idx % monitor_gap == 0:
                percent = get_ymir_process(stage=YmirStage.TASK, p=idx / N,
                                           task_idx=self.task_idx, task_num=self.task_num)
                monitor.write_monitor_logger(percent=percent)

        return mining_result

def main():
    cfg = get_merged_config()
    miner = MiningEntropy(cfg)
    mining_result = miner.mining()
    rw.write_mining_result(mining_result=mining_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())