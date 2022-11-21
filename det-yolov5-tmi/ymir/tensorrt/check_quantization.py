import os
import shutil
import argparse
import threading
import ctypes
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import numpy as np
from yolov5_trt import YoLov5TRT, warmUpThread, get_img_path_batches
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
        # shutil.rmtree('build/coco_calib')
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
            thread1 = warmUpThread(yolov5_wrapper)
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


def calculate_metrics(gt_dir, result_dir, image_dir):
    """
    param:
        gt_dir: ground truth labels files root directory, support multiple level
        result_dir: result labels files root directory, support multiple level
        image_dir: origin image files root directory, support multilpe level
    """
    gt_bbs = converter.text2bb(
        gt_dir,
        bb_type=BBType.GROUND_TRUTH,
        bb_format=BBFormat.YOLO,  # xcycwh
        type_coordinates=CoordinatesType.RELATIVE,
        img_dir=image_dir)

    det_bbs = converter.text2bb(result_dir,
                                bb_type=BBType.DETECTED,
                                bb_format=BBFormat.XYX2Y2,
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                img_dir=image_dir)

    #############################################################
    # EVALUATE WITH COCO METRICS
    #############################################################
    coco_res1 = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
    # coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)

    print('coco result \n')
    pprint(coco_res1)
    # pprint(coco_res2)
    #############################################################
    # EVALUATE WITH VOC PASCAL METRICS
    #############################################################
    ious = [0.5, 0.75]

    for iou in ious:
        dict_res = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs,
                                                              det_bbs,
                                                              iou,
                                                              generate_table=True,
                                                              method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        print(f'voc result, iou = {iou} \n')
        pprint(dict_res['mAP'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_number', type=int, required=True, help='the class number for model')
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'fp16', 'fp32'])
    parser.add_argument('--calibration_path', type=str, required=True, help='random select 1000 images for calibration')
    parser.add_argument('--weight_file', type=str, required=True, help='weight file')
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], help='the yolov5 model', required=True)
    parser.add_argument('--eval_image_path', type=str, required=True, help='images for eval')
    parser.add_argument('--gt_txt_dir', type=str, help='groundtruth txt file path')
    parser.add_argument('--yolov5_path', type=str, default='/app', help='the ultralytics yolov5 source code path')
    parser.add_argument('--conf_thresh', type=float, default=0.001)
    parser.add_argument('--iou_thresh', type=float, default=0.6)
    return parser.parse_args()


def main():
    args = get_args()
    cmake_build(args.class_number, dtype=args.dtype, nms_thresh=args.iou_thresh, conf_thresh=args.conf_thresh)

    engine_file = str(Path(args.weight_file).with_suffix('.engine'))
    convert_model(args.weight_file, engine_file, args.model, args.calibration_path, args.yolov5_path)

    result_dir = 'tmp_output'
    check_engine(engine_file,
               args.eval_image_path,
               result_dir=result_dir,
               conf_thresh=args.conf_thresh,
               iou_thresh=args.iou_thresh)

    calculate_metrics(gt_dir=args.gt_txt_dir, result_dir=result_dir, image_dir=args.eval_image_path)


if __name__ == '__main__':
    main()
