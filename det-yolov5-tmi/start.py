import json
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Dict, List

import cv2
from easydict import EasyDict as edict
from models.experimental import attempt_download
from utils.ymir_yolov5 import YmirYolov5, convert_ymir_to_yolov5, get_weight_file
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, find_free_port, get_bool, get_merged_config, get_ymir_process,
                           write_ymir_training_result)


def start(cfg: edict) -> int:
    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    else:
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            mining_task_idx = 0
            infer_task_idx = 1
            task_num = 2
        else:
            mining_task_idx = 0
            infer_task_idx = 0
            task_num = 1

        if cfg.ymir.run_mining:
            _run_mining(cfg, mining_task_idx, task_num)
        if cfg.ymir.run_infer:
            _run_infer(cfg, infer_task_idx, task_num)

    return 0


def _run_training(cfg: edict) -> None:
    """
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    # 1. convert dataset
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    # 2. training model
    epochs: int = int(cfg.param.epochs)
    batch_size_per_gpu: int = int(cfg.param.batch_size_per_gpu)
    num_workers_per_gpu: int = int(cfg.param.get('num_workers_per_gpu', 8))
    model: str = cfg.param.model
    img_size: int = int(cfg.param.img_size)
    save_period: int = max(1, min(epochs // 10, int(cfg.param.save_period)))
    args_options: str = cfg.param.args_options
    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
    batch_size: int = batch_size_per_gpu * max(1, gpu_count)
    port: int = int(cfg.param.get('port', 29500))
    sync_bn: bool = get_bool(cfg, key='sync_bn', default_value=False)

    weights = get_weight_file(cfg)
    if not weights:
        # download pretrained weight
        weights = attempt_download(f'{model}.pt')

    models_dir = cfg.ymir.output.models_dir
    project = os.path.dirname(models_dir)
    name = os.path.basename(models_dir)
    assert os.path.join(project, name) == models_dir

    commands = ['python3']
    device = gpu_id or 'cpu'
    if gpu_count > 1:
        commands.extend(f'-m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port}'.split())

    commands.extend([
        'train.py', '--epochs',
        str(epochs), '--batch-size',
        str(batch_size), '--data', f'{out_dir}/data.yaml', '--project', project, '--cfg', f'models/{model}.yaml',
        '--name', name, '--weights', weights, '--img-size',
        str(img_size), '--save-period',
        str(save_period), '--device', device, '--workers',
        str(num_workers_per_gpu)
    ])

    if gpu_count > 1 and sync_bn:
        commands.append("--sync-bn")

    if args_options:
        commands.extend(args_options.split())

    logging.info(f'start training: {commands}')

    subprocess.run(commands, check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # 3. convert to onnx and save model weight to design directory
    opset = cfg.param.opset
    command = f'python3 export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logging.info(f'export onnx weight: {command}')
    subprocess.run(command.split(), check=True)

    # 4. add attachments for quantification
    attachments: Dict[str, List[str]] = dict()
    attachments['images'] = []
    attachments['configs'] = []
    with open(cfg.ymir.input.val_index_file, 'r') as fp:
        img_files = [line.split()[0] for line in fp.readlines()]

    attachments_image_dir = osp.join(models_dir, 'attachments/images')
    os.makedirs(attachments_image_dir, exist_ok=True)
    for img_f in img_files[0:200]:
        shutil.copy(img_f, attachments_image_dir)
        attachments['images'].append(osp.basename(img_f))

    attachments_config_dir = osp.join(models_dir, 'attachments/configs')
    os.makedirs(attachments_config_dir, exist_ok=True)
    for config_f in ['preconfig.json', 'postconfig.json']:
        with open(config_f, 'r') as fp:
            quant_config = json.load(fp)

        if config_f == 'preconfig.json':
            quant_config['inputs'][0]['dims'] = [1, 3, img_size, img_size]
        else:
            from models.experimental import attempt_load  # scoped to avoid circular import
            quant_model = attempt_load(f'{models_dir}/best.pt', map_location='cpu')
            anchors = quant_model.model[-1].anchors * quant_model.stride.view(-1, 1, 1)
            np_anchors = anchors.data.cpu().numpy().reshape(3, -1).astype(int)
            quant_config['anchor0'] = np_anchors[0].tolist()
            quant_config['anchor0'] = np_anchors[1].tolist()
            quant_config['anchor0'] = np_anchors[2].tolist()

        out_config_f = osp.join(attachments_config_dir, config_f)
        with open(out_config_f, 'w') as fw:
            json.dump(quant_config, fw)

        # save to yaml with relative path to mdoels_dir
        attachments['configs'].append(osp.basename(config_f))

    write_ymir_training_result(cfg, map50=0, files=[], id='last', attachments=attachments)
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    # generate data.yaml for mining
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))
    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0

    mining_algorithm = cfg.param.get('mining_algorithm', 'aldd')
    support_mining_algorithms = ['aldd', 'cald']
    if mining_algorithm not in support_mining_algorithms:
        raise Exception(f'unknown mining algorithm {mining_algorithm}, not in {support_mining_algorithms}')

    if gpu_count <= 1 and mining_algorithm in ['cald']:
        command = 'python3 mining/mining_cald.py'
    else:
        port = find_free_port()
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port} mining/ymir_mining_{mining_algorithm}.py'  # noqa

    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.POSTPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))


def _run_infer(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    # generate data.yaml for infer
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))

    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0

    if gpu_count <= 1:
        N = dr.items_count(env.DatasetType.CANDIDATE)
        infer_result = dict()
        model = YmirYolov5(cfg)
        idx = -1

        monitor_gap = max(1, N // 100)
        for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
            img = cv2.imread(asset_path)
            result = model.infer(img)
            infer_result[asset_path] = result
            idx += 1

            if idx % monitor_gap == 0:
                percent = get_ymir_process(stage=YmirStage.TASK, p=idx / N, task_idx=task_idx, task_num=task_num)
                monitor.write_monitor_logger(percent=percent)

        rw.write_infer_result(infer_result=infer_result)
    else:
        port = find_free_port()
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port} mining/ymir_infer.py'  # noqa

        logging.info(f'infer: {command}')
        subprocess.run(command.split(), check=True)

    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.POSTPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    # activation: relu
    activation: str = cfg.param.get('activation', '')
    if activation:
        os.environ.setdefault('ACTIVATION', activation)
    sys.exit(start(cfg))
