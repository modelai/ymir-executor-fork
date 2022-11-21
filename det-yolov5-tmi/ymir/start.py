import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc import monitor
from ymir_exc.util import YmirStage, find_free_port, get_bool, get_merged_config, write_ymir_monitor_process

from models.experimental import attempt_download
from ymir.ymir_yolov5 import convert_ymir_to_yolov5, get_weight_file


def start(cfg: edict) -> int:
    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    else:
        if cfg.ymir.run_mining:
            _run_mining(cfg)
        if cfg.ymir.run_infer:
            _run_infer(cfg)

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
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    # 2. training model
    epochs: int = int(cfg.param.epochs)
    batch_size_per_gpu: int = int(cfg.param.batch_size_per_gpu)
    num_workers_per_gpu: int = int(cfg.param.get('num_workers_per_gpu', 4))
    model: str = cfg.param.model
    img_size: int = int(cfg.param.img_size)
    save_period: int = int(cfg.param.save_period)
    save_best_only: bool = get_bool(cfg, key='save_best_only', default_value=True)
    args_options: str = cfg.param.args_options
    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
    batch_size: int = batch_size_per_gpu * max(1, gpu_count)
    port: int = find_free_port()
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

    if save_best_only:
        commands.append("--nosave")

    if gpu_count > 1 and sync_bn:
        commands.append("--sync-bn")

    if args_options:
        commands.extend(args_options.split())

    logging.info(f'start training: {commands}')

    subprocess.run(commands, check=True)
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.TASK)

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict) -> None:
    # generate data.yaml for mining
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)
    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0

    mining_algorithm = cfg.param.get('mining_algorithm', 'aldd')
    support_mining_algorithms = ['aldd', 'cald', 'random', 'entropy']
    if mining_algorithm not in support_mining_algorithms:
        raise Exception(f'unknown mining algorithm {mining_algorithm}, not in {support_mining_algorithms}')

    if gpu_count <= 1:
        command = f'python3 ymir/mining/ymir_mining_{mining_algorithm}.py'
    else:
        port = find_free_port()
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port} ymir/mining/ymir_mining_{mining_algorithm}.py'  # noqa

    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


def _run_infer(cfg: edict) -> None:
    # generate data.yaml for infer
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0

    if gpu_count <= 1:
        command = 'python3 ymir/mining/ymir_infer.py'
    else:
        port = find_free_port()
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port} ymir/mining/ymir_infer.py'  # noqa

    logging.info(f'infer: {command}')
    subprocess.run(command.split(), check=True)

    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


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
