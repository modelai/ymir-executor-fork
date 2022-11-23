"""
ymir use tensorrt to check int8 quantization

1. link attachments_image_dir to build/coco_calib, 200 images will be copied to attachments_image_dir after training
2. parse val_index_file to obtain full validation images
3. parse val_index_file to obtain full ymir-format(class_id xyxy int absolute) annotation files
4. compute the map for coco and voc and write to result.yaml
"""
import os.path as osp
import sys
from ymir_exc.util import get_merged_config
from ymir.ymir_yolov5 import get_attachments
from ymir_exc.util import write_ymir_training_result
from ymir_exc import monitor
import subprocess
import shutil
from easydict import EasyDict as edict
import os


def generate_fake_result_file(ymir_cfg: edict, fake_map=0.9):
    """
    just for convert, not run training
    """
    attachments = get_attachments(ymir_cfg)

    os.makedirs('/out/models', exist_ok=True)
    shutil.copy('/in/models/best.onnx', '/out/models/best.onnx')
    shutil.copy('/in/models/best.pt', '/out/models/best.pt')
    write_ymir_training_result(ymir_cfg,
                               map50=fake_map,
                               id='best',
                               files=['/out/models/best.onnx', '/out/models/best.pt'],
                               attachments=attachments)


def main() -> int:
    cfg = get_merged_config()
    model = cfg.param.model

    result_file = cfg.ymir.output.training_result_file
    if not osp.exists(result_file):
        generate_fake_result_file(cfg)

    # yolov5n/s/m/l/x --> n/s/m/l/x
    short_model = model[6:]
    class_number = len(cfg.param.class_names)
    models_dir: str = cfg.ymir.output.models_dir
    attachments_image_dir = osp.join(models_dir, 'attachments/images')

    assert osp.exists(
        attachments_image_dir), f'please train model and generate attachments first, {attachments_image_dir} not exists'

    weight_file = osp.join(models_dir, 'best.pt')
    batch_size = int(cfg.param.batch_size_per_gpu)
    workers = int(cfg.param.num_workers_per_gpu)
    conf_thresh = 0.1
    iou_thresh = 0.6

    # yolov5 validation, save txt to /app/ymir/tensorrt/pt_result/labels
    pt_result_dir = 'ymir/tensorrt/pt_result/labels'
    # yolov5 generate result file with file 'a' mode
    if osp.isdir(pt_result_dir):
        print(f'remove yolov5 pytorch result directory {pt_result_dir}')
        shutil.rmtree(pt_result_dir)

    pt_cmd = [
        'python3', 'val.py', '--weights', weight_file, '--data', '/out/data.yaml', '--batch-size',
        str(batch_size), '--workers',
        str(workers), '--conf-thres',
        str(conf_thresh), '--iou-thres',
        str(iou_thresh), '--save-txt', '--save-conf', '--project', 'ymir/tensorrt', '--name', 'pt_result', '--exist-ok'
    ]
    print(f'run commands: {pt_cmd}')
    subprocess.run(pt_cmd, check=True)

    # ymir txt annotation format: class_id xyxy absolute
    val_index_file = cfg.ymir.input.val_index_file
    yolov5_path = '/app'

    trt_cmd = [
        'python3', 'check_quantization.py', '--class_number',
        str(class_number), '--dtype', 'int8', '--calibration_path', attachments_image_dir, '--weight_file', weight_file,
        '--model', short_model, '--ymir_eval_index_file', val_index_file, '--yolov5_path', yolov5_path, '--conf_thresh',
        str(conf_thresh), '--iou_thresh',
        str(iou_thresh), '--pt_result_dir', 'pt_result/labels'
    ]

    print(f'run commands: {trt_cmd}')
    subprocess.run(trt_cmd, check=True, cwd='ymir/tensorrt')

    monitor.write_monitor_logger(percent=1.0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
