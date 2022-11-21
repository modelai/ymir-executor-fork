import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc import monitor
from ymir_exc.util import find_free_port, get_merged_config


def start(cfg: edict) -> int:
    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training()
    elif cfg.ymir.run_mining or cfg.ymir.run_infer:
        if cfg.ymir.run_mining:
            _run_mining(cfg)
        if cfg.ymir.run_infer:
            _run_infer()
    else:
        logging.warning('no task running')

    return 0


def _run_training() -> None:
    command = 'python3 ymir_train.py'
    logging.info(f'start training: {command}')
    subprocess.run(command.split(), check=True)

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)
    logging.info("training finished")


def _run_mining(cfg: edict) -> None:
    gpu_id: str = str(cfg.param.get('gpu_id', '0'))
    gpu_count = len(gpu_id.split(','))
    mining_algorithm: str = cfg.param.get('mining_algorithm', 'aldd')

    supported_miner = ['cald', 'aldd', 'random', 'entropy']
    assert mining_algorithm in supported_miner, f'unknown mining_algorithm {mining_algorithm}, not in {supported_miner}'
    if gpu_count <= 1:
        command = f'python3 ymir_mining_{mining_algorithm}.py'
    else:
        port = find_free_port()
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port} ymir_mining_{mining_algorithm}.py'  # noqa

    logging.info(f'start mining: {command}')
    subprocess.run(command.split(), check=True)
    logging.info("mining finished")


def _run_infer() -> None:
    command = 'python3 ymir_infer.py'
    logging.info(f'start infer: {command}')
    subprocess.run(command.split(), check=True)
    logging.info("infer finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    cfg = get_merged_config()
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    os.environ.setdefault('COCO_EVAL_TMP_FILE', os.path.join(cfg.ymir.output.root_dir, 'eval_tmp.json'))
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start(cfg))
