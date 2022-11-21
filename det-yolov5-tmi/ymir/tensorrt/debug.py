from object_detection_metrics.utils import converter
from object_detection_metrics.evaluators import coco_evaluator, pascal_voc_evaluator
from object_detection_metrics.utils.enumerators import (BBFormat, BBType, CoordinatesType, MethodAveragePrecision)
from pprint import pprint

gt_dir = '/out/labels/val'
result_dir = 'tmp_output'
image_dir = '/in/assets/val'

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

print('gt:', len(gt_bbs), 'det:', len(det_bbs))
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
