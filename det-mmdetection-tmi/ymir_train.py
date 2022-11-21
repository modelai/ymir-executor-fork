import logging
import os
import os.path as osp
import subprocess
import sys

from easydict import EasyDict as edict
from mmdet.utils.util_ymir import get_best_weight_file, write_ymir_training_result
from ymir_exc import monitor
from ymir_exc.util import YmirStage, find_free_port, get_merged_config, write_ymir_monitor_process


def main(cfg: edict) -> int:
    # default ymir config
    gpu_id: str = str(cfg.param.get("gpu_id", '0'))
    num_gpus = len(gpu_id.split(","))
    if num_gpus == 0:
        raise Exception(f'gpu_id = {gpu_id} is not valid, eg: 0 or 2,4')

    classes = cfg.param.class_names
    num_classes = len(classes)
    if num_classes == 0:
        raise Exception('not find class_names in config!')

    # mmcv args config
    config_file = cfg.param.get("config_file")
    args_options = cfg.param.get("args_options", None)
    cfg_options = cfg.param.get("cfg_options", None)

    # auto load offered weight file if not set by user!
    if (args_options is None or args_options.find('--resume-from') == -1) and \
            (cfg_options is None or (cfg_options.find('load_from') == -1 and
                                     cfg_options.find('resume_from') == -1)):

        weight_file = get_best_weight_file(cfg)
        if weight_file:
            if cfg_options:
                cfg_options += f' load_from={weight_file}'
            else:
                cfg_options = f'load_from={weight_file}'
        else:
            logging.warning('no weight file used for training!')

    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=0.2, stage=YmirStage.POSTPROCESS)

    work_dir = cfg.ymir.output.models_dir
    if num_gpus == 0:
        # view https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#training-on-cpu
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', "-1")
        cmd = f"python3 tools/train.py {config_file} " + \
            f"--work-dir {work_dir}"
    elif num_gpus == 1:
        cmd = f"python3 tools/train.py {config_file} " + \
            f"--work-dir {work_dir} --gpu-id {gpu_id}"
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        port = find_free_port()
        os.environ.setdefault('PORT', str(port))
        cmd = f"bash ./tools/dist_train.sh {config_file} {num_gpus} " + \
            f"--work-dir {work_dir}"

    if args_options:
        cmd += f" {args_options}"

    if cfg_options:
        cmd += f" --cfg-options {cfg_options}"

    logging.info(f"training command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    # save the last checkpoint
    write_ymir_training_result(last=True)
    return 0


if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    os.environ.setdefault('COCO_EVAL_TMP_FILE', osp.join(
        cfg.ymir.output.root_dir, 'eval_tmp.json'))
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(main(cfg))
