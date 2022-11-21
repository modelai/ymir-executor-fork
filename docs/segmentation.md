# segmentation

- update date: 2022/11/14

## semantic segmentation: 语义分割

### docker images: docker 镜像

- `youdaoyzbx/ymir-executor:ymir2.0.0-mmseg-cu111-tmi`

### hyper-parameters: 超参数

- training: 训练
    - `export_format`: `seg-mask:raw`

### convert dataset format: 转换数据集格式

```
from ymir_exc.dataset_convert import convert_ymir_to_mmseg
from ymir_exc.util import get_merged_config

ymir_cfg = get_merged_config()
new_ann_dict = convert_ymir_to_mmseg(ymir_cfg)
```

### read: 输入格式

```
in
├── annotations [19 entries exceeds filelimit, not opening dir]
├── assets -> /xxx/ymir-workplace/sandbox/0001/asset_cache
├── config.yaml
├── env.yaml
├── idx-assets.tsv
├── idx-gt.tsv
├── idx-pred.tsv
├── models
├── predictions [18 entries exceeds filelimit, not opening dir]
├── pred-test-index.tsv
├── pred-train-index.tsv
├── pred-val-index.tsv
├── test-index.tsv
├── train-index.tsv
└── val-index.tsv
```

## in/annotations
```
ls /in/annotations
08  15  19  32  35  3b  59  6a  72  77  85  a4  a6  cd  d1  e0  e1  f0  labelmap.txt
```

## in/annotations/labelmap.txt

- `class_name:R,G,B::`:
    - class_name=bg, RGB=(0, 0, 0)
    - class_name=fg, RGB=(1, 1, 1)

- `cat in/annotations/labelmap.txt`
```
bg:0,0,0::
fg:1,1,1::
```

## in/env.yaml
```
input:                                                                                            [2/1804]
  annotations_dir: /in/annotations
  assets_dir: /in/assets
  candidate_index_file: ''
  config_file: /in/config.yaml
  models_dir: /in/models
  root_dir: /in
  training_index_file: /in/train-index.tsv
  val_index_file: /in/val-index.tsv
output:
  executor_log_file: /out/ymir-executor-out.log
  infer_result_file: /out/infer-result.json
  mining_result_file: /out/result.tsv
  models_dir: /out/models
  monitor_file: /out/monitor.txt
  root_dir: /out
  tensorboard_dir: /out/tensorboard
  training_result_file: /out/models/result.yaml
protocol_version: 1.1.0
run_infer: false
run_mining: false
run_training: true
task_id: t00000010000059a17ce1668392602
```

## in/train-index.tsv
```
/in/assets/32/6371cbb7e0a2c356cb17e17ca467c7f892ccc232.png      /in/annotations/32/6371cbb7e0a2c356cb17e17ca467c7f892ccc232.png
/in/assets/32/562cfd8c96bba98568673d59614d2578258f1e32.png      /in/annotations/32/562cfd8c96bba98568673d59614d2578258f1e32.png
/in/assets/59/f72430463f59d0299c3258e01fc9ad2c5671b359.png      /in/annotations/59/f72430463f59d0299c3258e01fc9ad2c5671b359.png
```

### write: 输出格式
```
out
├── models [17 entries exceeds filelimit, not opening dir]
├── monitor.txt
├── tensorboard -> /xxx/ymir-workplace/ymir-tensorboard-logs/0001/t00000010000059a17ce1668392602
└── ymir-executor-out.log
```

- `ls /out/models`
```
20221114_022352.log                      iter_1000.pth  iter_1800.pth  iter_600.pth  ymir-info.yaml
20221114_022352.log.json                 iter_1200.pth  iter_2000.pth  iter_800.pth
best_mIoU_iter_1200.pth                  iter_1400.pth  iter_200.pth   latest.pth
fast_scnn_lr0.12_8x4_160k_cityscapes.py  iter_1600.pth  iter_400.pth   result.yaml
```

- `cat /out/models/result.yaml`
```
best_stage_name: best
map: 0.632
model_stages:
  best:
    files:
    - fast_scnn_lr0.12_8x4_160k_cityscapes.py
    - best_mIoU_iter_1200.pth
    mAP: 0.632
    stage_name: best
    timestamp: 1668393850
  last:
    files:
    - fast_scnn_lr0.12_8x4_160k_cityscapes.py
    - latest.pth
    mAP: 0.5421
    stage_name: last
    timestamp: 1668393874
```

## instance segmentation: 实例分割
todo: 开发中
