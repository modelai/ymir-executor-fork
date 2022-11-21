"""
utils function for ymir and yolov5
"""
import glob
import logging
import os
import os.path as osp
from typing import Any, Iterable, List, Optional, Union

import mmcv
import yaml
from easydict import EasyDict as edict
from mmcv import Config, ConfigDict
from nptyping import NDArray, Shape, UInt8
from packaging.version import Version
from ymir_exc import result_writer as rw
from ymir_exc.util import get_merged_config

BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def modify_mmcv_config(mmcv_cfg: Config, ymir_cfg: edict) -> None:
    """
    useful for training process
    - modify dataset config
    - modify model output channel
    - modify epochs, checkpoint, tensorboard config
    """

    def recursive_modify_attribute(mmcv_cfgdict: Union[Config, ConfigDict], attribute_key: str, attribute_value: Any):
        """
        recursive modify mmcv_cfg:
            1. mmcv_cfg.attribute_key to attribute_value
            2. mmcv_cfg.xxx.xxx.xxx.attribute_key to attribute_value (recursive)
            3. mmcv_cfg.xxx[i].attribute_key to attribute_value (i=0, 1, 2 ...)
            4. mmcv_cfg.xxx[i].xxx.xxx[j].attribute_key to attribute_value
        """
        for key in mmcv_cfgdict:
            if key == attribute_key:
                mmcv_cfgdict[key] = attribute_value
                logging.info(f'modify {mmcv_cfgdict}, {key} = {attribute_value}')
            elif isinstance(mmcv_cfgdict[key], (Config, ConfigDict)):
                recursive_modify_attribute(mmcv_cfgdict[key], attribute_key, attribute_value)
            elif isinstance(mmcv_cfgdict[key], Iterable):
                for cfg in mmcv_cfgdict[key]:
                    if isinstance(cfg, (Config, ConfigDict)):
                        recursive_modify_attribute(cfg, attribute_key, attribute_value)

    # modify dataset config
    ymir_ann_files = dict(train=ymir_cfg.ymir.input.training_index_file,
                          val=ymir_cfg.ymir.input.val_index_file,
                          test=ymir_cfg.ymir.input.candidate_index_file)

    # validation may augment the image and use more gpu
    # so set smaller samples_per_gpu for validation
    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    mmcv_cfg.data.samples_per_gpu = samples_per_gpu
    mmcv_cfg.data.workers_per_gpu = workers_per_gpu

    # modify model output channel
    num_classes = len(ymir_cfg.param.class_names)
    recursive_modify_attribute(mmcv_cfg.model, 'num_classes', num_classes)

    for split in ['train', 'val', 'test']:
        ymir_dataset_cfg = dict(type='YmirDataset',
                                ann_file=ymir_ann_files[split],
                                img_prefix=ymir_cfg.ymir.input.assets_dir,
                                ann_prefix=ymir_cfg.ymir.input.annotations_dir,
                                classes=ymir_cfg.param.class_names,
                                data_root=ymir_cfg.ymir.input.root_dir,
                                filter_empty_gt=False)
        # modify dataset config for `split`
        mmdet_dataset_cfg = mmcv_cfg.data.get(split, None)
        if mmdet_dataset_cfg is None:
            continue

        if isinstance(mmdet_dataset_cfg, (list, tuple)):
            for x in mmdet_dataset_cfg:
                x.update(ymir_dataset_cfg)
        else:
            src_dataset_type = mmdet_dataset_cfg.type
            if src_dataset_type in ['CocoDataset', 'YmirDataset']:
                mmdet_dataset_cfg.update(ymir_dataset_cfg)
            elif src_dataset_type in ['MultiImageMixDataset', 'RepeatDataset']:
                mmdet_dataset_cfg.dataset.update(ymir_dataset_cfg)
            else:
                raise Exception(f'unsupported source dataset type {src_dataset_type}')

    # modify epochs, checkpoint, tensorboard config
    if ymir_cfg.param.get('max_epochs', None):
        mmcv_cfg.runner.max_epochs = int(ymir_cfg.param.max_epochs)
    mmcv_cfg.checkpoint_config['out_dir'] = ymir_cfg.ymir.output.models_dir
    tensorboard_logger = dict(type='TensorboardLoggerHook', log_dir=ymir_cfg.ymir.output.tensorboard_dir)
    if len(mmcv_cfg.log_config['hooks']) <= 1:
        mmcv_cfg.log_config['hooks'].append(tensorboard_logger)
    else:
        mmcv_cfg.log_config['hooks'][1].update(tensorboard_logger)

    # TODO save only the best top-k model weight files.
    # modify evaluation and interval
    val_interval: int = int(ymir_cfg.param.get('val_interval', 1))
    if val_interval > 0:
        val_interval = min(val_interval, mmcv_cfg.runner.max_epochs)
    else:
        val_interval = 1

    mmcv_cfg.evaluation.interval = val_interval
    mmcv_cfg.evaluation.metric = ymir_cfg.param.get('metric', 'bbox')

    # save best top-k model weights files
    # max_keep_ckpts <= 0  # save all checkpoints
    max_keep_ckpts: int = int(ymir_cfg.param.get('max_keep_checkpoints', 1))
    mmcv_cfg.checkpoint_config.interval = mmcv_cfg.evaluation.interval
    mmcv_cfg.checkpoint_config.max_keep_ckpts = max_keep_ckpts

    # TODO Whether to evaluating the AP for each class
    # mmdet_cfg.evaluation.classwise = True

    # fix DDP error
    mmcv_cfg.find_unused_parameters = True

    # set work dir
    mmcv_cfg.work_dir = ymir_cfg.ymir.output.models_dir

    args_options = ymir_cfg.param.get("args_options", '')
    cfg_options = ymir_cfg.param.get("cfg_options", '')

    # auto load offered weight file if not set by user!
    if (args_options.find('--resume-from') == -1 and args_options.find('--load-from') == -1
            and cfg_options.find('load_from') == -1 and cfg_options.find('resume_from') == -1):  # noqa: E129

        weight_file = get_best_weight_file(ymir_cfg)
        if weight_file:
            if cfg_options:
                cfg_options += f' load_from={weight_file}'
            else:
                cfg_options = f'load_from={weight_file}'
        else:
            logging.warning('no weight file used for training!')


def get_best_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.pretrained_model_params or cfg.param.model_params_path
    load coco-pretrained weight for yolox
    """
    if cfg.ymir.run_training:
        model_params_path: List[str] = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path = cfg.param.get('model_params_path', [])

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path
        if osp.exists(osp.join(model_dir, p)) and p.endswith(('.pth', '.pt'))
    ]

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [f for f in model_params_path if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)

    epoch_pth_files = [f for f in model_params_path if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    if cfg.ymir.run_training:
        weight_files = [f for f in glob.glob('/weights/**/*', recursive=True) if f.endswith(('.pth', '.pt'))]

        # load pretrained model weight for yolox only
        model_name_splits = osp.basename(cfg.param.config_file).split('_')
        if len(weight_files) > 0 and model_name_splits[0] == 'yolox':
            yolox_weight_files = [
                f for f in weight_files if osp.basename(f).startswith(f'yolox_{model_name_splits[1]}')
            ]

            if len(yolox_weight_files) == 0:
                if model_name_splits[1] == 'nano':
                    # yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth or yolox_tiny.py
                    yolox_weight_files = [f for f in weight_files if osp.basename(f).startswith('yolox_tiny')]
                else:
                    yolox_weight_files = [f for f in weight_files if osp.basename(f).startswith('yolox_s')]

            if len(yolox_weight_files) > 0:
                logging.info(f'load yolox pretrained weight {yolox_weight_files[0]}')
                return yolox_weight_files[0]
    return ""


def write_ymir_training_result(last: bool = False, key_score: Optional[float] = None):
    YMIR_VERSION = os.environ.get('YMIR_VERSION', '1.2.0')
    if Version(YMIR_VERSION) >= Version('1.2.0'):
        _write_latest_ymir_training_result(last, key_score)
    else:
        _write_ancient_ymir_training_result(key_score)


def get_topk_checkpoints(files: List[str], k: int) -> List[str]:
    """
    keep topk checkpoint files, remove other files.
    """
    checkpoints_files = [f for f in files if f.endswith(('.pth', '.pt'))]

    best_pth_files = [f for f in checkpoints_files if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        # newest first
        topk_best_pth_files = sorted(best_pth_files, key=os.path.getctime, reverse=True)
    else:
        topk_best_pth_files = []

    epoch_pth_files = [f for f in checkpoints_files if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        topk_epoch_pth_files = sorted(epoch_pth_files, key=os.path.getctime, reverse=True)
    else:
        topk_epoch_pth_files = []

    # python will check the length of list
    return topk_best_pth_files[0:k] + topk_epoch_pth_files[0:k]


# TODO save topk checkpoints, fix invalid stage due to delete checkpoint
def _write_latest_ymir_training_result(last: bool = False, key_score: Optional[float] = None):
    if key_score:
        logging.info(f'key_score is {key_score}')
    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception('please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    eval_result = mmcv.load(COCO_EVAL_TMP_FILE)
    # eval_result may be empty dict {}.
    map = eval_result.get('bbox_mAP_50', 0)

    WORK_DIR = os.getenv('YMIR_MODELS_DIR')
    if WORK_DIR is None or not osp.isdir(WORK_DIR):
        raise Exception(f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {WORK_DIR}')

    # assert only one model config file in work_dir
    result_files = [f for f in glob.glob(osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml']

    if last:
        # save all output file
        ymir_cfg = get_merged_config()
        max_keep_checkpoints = int(ymir_cfg.param.get('max_keep_checkpoints', 1))
        if max_keep_checkpoints > 0:
            topk_checkpoints = get_topk_checkpoints(result_files, max_keep_checkpoints)
            result_files = [f for f in result_files if not f.endswith(('.pth', '.pt'))] + topk_checkpoints

        result_files = [osp.basename(f) for f in result_files]
        rw.write_model_stage(files=result_files, mAP=float(map), stage_name='last')
    else:
        result_files = [osp.basename(f) for f in result_files]
        # save newest weight file in format epoch_xxx.pth or iter_xxx.pth
        weight_files = [
            osp.join(WORK_DIR, f) for f in result_files if f.startswith(('iter_', 'epoch_')) and f.endswith('.pth')
        ]

        if len(weight_files) > 0:
            newest_weight_file = osp.basename(max(weight_files, key=os.path.getctime))

            stage_name = osp.splitext(newest_weight_file)[0]
            training_result_file = osp.join(WORK_DIR, 'result.yaml')
            if osp.exists(training_result_file):
                with open(training_result_file, 'r') as f:
                    training_result = yaml.safe_load(f)
                    model_stages = training_result.get('model_stages', {})
            else:
                model_stages = {}

            if stage_name not in model_stages:
                config_files = [f for f in result_files if f.endswith('.py')]
                rw.write_model_stage(files=[newest_weight_file] + config_files, mAP=float(map), stage_name=stage_name)


def _write_ancient_ymir_training_result(key_score: Optional[float] = None):
    if key_score:
        logging.info(f'key_score is {key_score}')

    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception('please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    eval_result = mmcv.load(COCO_EVAL_TMP_FILE)
    # eval_result may be empty dict {}.
    map = eval_result.get('bbox_mAP_50', 0)

    ymir_cfg = get_merged_config()
    WORK_DIR = ymir_cfg.ymir.output.models_dir

    # assert only one model config file in work_dir
    result_files = [f for f in glob.glob(osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml']

    max_keep_checkpoints = int(ymir_cfg.param.get('max_keep_checkpoints', 1))
    if max_keep_checkpoints > 0:
        topk_checkpoints = get_topk_checkpoints(result_files, max_keep_checkpoints)
        result_files = [f for f in result_files if not f.endswith(('.pth', '.pt'))] + topk_checkpoints

    # convert to basename
    result_files = [osp.basename(f) for f in result_files]

    training_result_file = osp.join(WORK_DIR, 'result.yaml')
    if osp.exists(training_result_file):
        with open(training_result_file, 'r') as f:
            training_result = yaml.safe_load(f)

        training_result['model'] = result_files
        training_result['map'] = max(map, training_result['map'])
    else:
        training_result = dict(model=result_files, map=map)

    with open(training_result_file, 'w') as f:
        yaml.safe_dump(training_result, f)
