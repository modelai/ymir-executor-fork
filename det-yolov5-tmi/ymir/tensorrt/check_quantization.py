import os
import shutil
import argparse
import threading
import ctypes
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import numpy as np
import yaml
from yolov5_trt import YoLov5TRT, WarmUpThread, get_img_path_batches
from object_detection_metrics.utils import converter
from object_detection_metrics.evaluators import coco_evaluator, pascal_voc_evaluator
from object_detection_metrics.utils.enumerators import (BBFormat, BBType, CoordinatesType, MethodAveragePrecision)


class EvalThread(threading.Thread):

    def __init__(self, yolov5_wrapper, image_path_batch, result_dir='output', conf_thresh=0.5, iou_thresh=0.4):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch
        self.result_dir = result_dir
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def run(self):
        results = {}
        for batch in tqdm(self.image_path_batch):
            batch_result, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(batch),
                                                               plot=False,
                                                               conf_thresh=self.conf_thresh,
                                                               iou_thresh=self.iou_thresh)

            for i, img_path in enumerate(batch):
                parent, filename = os.path.split(img_path)
                txt_name = str(Path(filename).with_suffix('.txt'))
                save_name = os.path.join(self.result_dir, txt_name)
                # Save result
                results[save_name] = batch_result[i]
                with open(save_name, 'w') as fp:
                    bboxes: np.ndarray = batch_result[i]['bboxes']  # [x1, y1, x2, y2]
                    scores: np.ndarray = batch_result[i]['scores']
                    class_ids: np.ndarray = batch_result[i]['class_ids']
                    for box, score, id in zip(bboxes, scores, class_ids):
                        x1, y1, x2, y2 = box
                        fp.write(f'{id} {score} {x1} {y1} {x2} {y2}\n')  # classid_abs_xyx2y2


def cmake_build(class_number, dtype='int8', nms_thresh=0.4, conf_thresh=0.5, batch_size=1, device=0):
    os.system("sed -i 's/CLASS_NUM = [0-9]\+/CLASS_NUM = {}/g' yololayer.h".format(class_number))  # noqa

    os.system("sed -i 's/#define USE_..../#define USE_{}/g' yolov5.cpp".format(dtype.upper()))  # noqa
    os.system("sed -i 's/#define NMS_THRESH [0-9.]\+/#define NMS_THRESH {}/g' yolov5.cpp".format(nms_thresh))  # noqa
    os.system("sed -i 's/#define CONF_THRESH [0-9.]\+/#define CONF_THRESH {}/g' yolov5.cpp".format(conf_thresh))  # noqa
    os.system("sed -i 's/#define BATCH_SIZE [0-9]\+/#define BATCH_SIZE {}/g' yolov5.cpp".format(batch_size))  # noqa
    os.system("sed -i 's/#define DEVICE [0-9]\+/#define DEVICE {}/g' yolov5.cpp".format(device))  # noqa

    os.makedirs('build', exist_ok=True)
    os.system("cd build && cmake .. && make -j4")


def convert_model(weight_file, engine_file, model, calibration_path, yolov5_path):
    assert model in 'nsmlx'
    wts_file = str(Path(weight_file).with_suffix('.wts'))
    os.system(f'cd {yolov5_path} && python ymir/tensorrt/gen_wts.py -w {weight_file} -o {wts_file}')
    if os.path.isdir('build/coco_calib'):
        os.unlink('build/coco_calib')
    os.system(f'cd build && ln -s {calibration_path} coco_calib')
    os.system(f'cd build && ./yolov5 -s {wts_file} {engine_file} {model}')


def check_engine(engine_file_path, image_dir, result_dir='tmp_output', conf_thresh=0.5, iou_thresh=0.4):
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    # engine_file_path = "build/yolov5n.engine"

    ctypes.CDLL(PLUGIN_LIBRARY)

    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)

    os.makedirs(result_dir)

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    try:
        print('batch size is', yolov5_wrapper.batch_size)
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = WarmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()

        # create a new thread to do inference
        thread1 = EvalThread(yolov5_wrapper,
                             image_path_batches,
                             result_dir=result_dir,
                             conf_thresh=conf_thresh,
                             iou_thresh=iou_thresh)
        thread1.start()
        thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()


def calculate_metrics(ymir_index_file, trt_result_dir, pt_result_dir):
    """
    param:
        ymir_index_file: ymir index file
        trt_result_dir: tensorrt result labels files root directory, support multiple level
        pt_result_dir: yolov5 result files root directory, support multilpe level
    """
    assert os.path.isfile(ymir_index_file), f'{ymir_index_file} is not ymir index file'

    with open(ymir_index_file) as fp:
        lines = fp.readlines()

    img_files = []
    img_to_ann = {}
    for line in lines:
        img_f, ann_f = line.split()
        img_files.append(img_f)

        result_f = os.path.join(pt_result_dir, os.path.basename(ann_f))
        if os.path.exists(result_f):
            img_to_ann[img_f] = result_f

    # ymir index file: class_id xyxy
    gt_bbs = converter.ymir2bb(ymir_index_file,
                               bb_type=BBType.GROUND_TRUTH,
                               bb_format=BBFormat.XYX2Y2,
                               type_coordinates=CoordinatesType.ABSOLUTE)

    # yolov5 tensorrt result
    trt_bbs = converter.text2bb(trt_result_dir,
                                bb_type=BBType.DETECTED,
                                bb_format=BBFormat.XYX2Y2,
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                img_dir=None)

    # yolov5 pytorch model result
    pt_bbs = converter.dict2bb(img_to_ann=img_to_ann,
                               bb_type=BBType.DETECTED,
                               bb_format=BBFormat.YOLO,
                               type_coordinates=CoordinatesType.RELATIVE)

    #############################################################
    # EVALUATE WITH COCO METRICS
    #############################################################
    coco_trt = coco_evaluator.get_coco_summary(gt_bbs, trt_bbs)
    # coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)
    coco_pt = coco_evaluator.get_coco_summary(gt_bbs, pt_bbs)

    print('coco result \n')
    pprint(coco_trt)
    pprint(coco_pt)
    # pprint(coco_res2)
    #############################################################
    # EVALUATE WITH VOC PASCAL METRICS
    #############################################################
    voc_trt = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs,
                                                         trt_bbs,
                                                         iou_threshold=0.5,
                                                         generate_table=False,
                                                         method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)

    voc_pt = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs,
                                                        pt_bbs,
                                                        iou_threshold=0.5,
                                                        generate_table=False,
                                                        method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
    print('tensorrt result: ', voc_trt['mAP'])
    print('yolov5 pytorch result: ', voc_pt['mAP'])

    result_file = '/out/models/result.yaml'

    with open(result_file, 'r') as fr:
        result = yaml.safe_load(fr)
    assert isinstance(result, dict), f'result {result} is not dict'

    eval_file = '/out/models/ymir_eval.yaml'
    eval_result = {}
    eval_result['voc_mAP_int8']: float = float(round(voc_trt['mAP'], ndigits=4))
    eval_result['voc_mAP_fp32']: float = float(round(voc_pt['mAP'], ndigits=4))

    with open(eval_file, 'w') as fw:
        yaml.safe_dump(eval_result, fw)

    # add eval_file to model stages
    for stage in result['model_stages']:
        result['model_stages'][stage]['files'].append(os.path.basename(eval_file))

    with open(result_file, 'w') as f:
        yaml.safe_dump(result, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_number', type=int, required=True, help='the class number for model')
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'fp16', 'fp32'])
    parser.add_argument('--calibration_path', type=str, required=True, help='random select 1000 images for calibration')
    parser.add_argument('--weight_file', type=str, required=True, help='weight file')
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], help='the yolov5 model', required=True)
    parser.add_argument('--ymir_eval_index_file', type=str, help='ymir validation index file')
    parser.add_argument('--pt_result_dir', type=str, help='the yolov5 validation output directory')
    parser.add_argument('--yolov5_path', type=str, default='/app', help='the ultralytics yolov5 source code path')
    parser.add_argument('--conf_thresh', type=float, default=0.001)
    parser.add_argument('--iou_thresh', type=float, default=0.6)
    return parser.parse_args()


def main():
    args = get_args()
    cmake_build(args.class_number, dtype=args.dtype, nms_thresh=args.iou_thresh, conf_thresh=args.conf_thresh)

    engine_file = str(Path(args.weight_file).with_suffix('.engine'))
    convert_model(args.weight_file, engine_file, args.model, args.calibration_path, args.yolov5_path)

    trt_result_dir = 'trt_result'
    check_engine(engine_file,
                 args.ymir_eval_index_file,
                 result_dir=trt_result_dir,
                 conf_thresh=args.conf_thresh,
                 iou_thresh=args.iou_thresh)

    pt_result_dir = 'pt_result/labels'
    calculate_metrics(ymir_index_file=args.ymir_eval_index_file,
                      trt_result_dir=trt_result_dir,
                      pt_result_dir=pt_result_dir)


if __name__ == '__main__':
    main()
