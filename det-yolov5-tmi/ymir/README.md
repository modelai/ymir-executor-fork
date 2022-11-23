# yolov5-ymir readme
update 2022/11/08

## build your ymir-executor

```
docker build -t your/ymir-executor:ymir2.0.0-cuda102-yolov5-tmi -f cuda102.dockerfile .

docker build -t your/ymir-executor:ymir2.0.0-cuda111-yolov5-tmi -f cuda111.dockerfile .

docker build -t your/ymir-executor:ymir2.0.0-yolov5-cpu-tmi -f cpu.dockerfile .
```

## 训练: training

### 性能表现

### 训练参数说明

- 一些参数由ymir后台生成，如 `gpu_index`, `class_names` 等参数
  - `gpu_index`:
  - `task_id`:
  - `model_params_path`:
  - `class_names`:

- 一些参数由ymir后台进行处理，如 `shm_size`, `export_format`， 其中 `shm_size` 影响到docker镜像所能使用的共享内存，若过小会导致 `out of memory` 等错误。 `export_format` 会决定docker镜像中所看到数据的格式



| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串：str | 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串：str | 受ymir后台处理，ymir数据集导出格式 | - |
| model | yolov5s | 字符串：str | yolov5模型，可选yolov5n, yolov5s, yolov5m, yolov5l等 | 建议：速度快选yolov5n, 精度高选yolov5l, yolov5x, 平衡选yolov5s或yolov5m |
| batch_size_per_gpu | 16 | 整数：int | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数：int | 每张GPU对应的数据读取进程数 | - |
| epochs | 100 | 整数：int | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| img_size | 640 | 整数: int | 输入模型的图像分辨率 | - |
| opset | 11 | 整数: int | onnx 导出参数 opset | 建议：一般不需要用到onnx，不必改 |
| args_options | '--exist-ok' | 字符串：str | yolov5命令行参数 | 建议：专业用户可用yolov5所有命令行参数 |
| save_best_only | True | 布尔: bool | 是否只保存最优模型 | 建议：为节省空间设为True即可 |
| save_period | 10 | 整数: int | 保存模型的间隔 | 建议：当save_best_only为False时，可保存 `epoch/save_period` 个中间结果
| sync_bn | False | 布尔: bool | 是否同步各gpu上的归一化层 | 建议：开启以提高训练稳定性及精度 |
| activate | '' | 字符串：str | 激活函数，默认为nn.Hardswish(), 参考 [pytorch激活函数](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) | 可选值: ELU, Hardswish, LeakyReLU, PReLU, ReLU, ReLU6, SiLU, ... |
| ymir_saved_file_patterns | '' | 字符串: str | 用 `,` 分隔的保存文件模式 | 建议：专业用户当希望过滤保存的文件以节省空间时，可设置配置的正则表达式 |

### 训练结果文件示例
```

```

## 推理: infer

### 推理参数说明

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |


### 推理结果文件示例

## main change log

- add `start.py` and `ymir/ymir_yolov5.py` for train/infer/mining

- add `ymir/ymir_yolov5.py` for useful functions

    - `get_merged_config()` add ymir path config `cfg.yaml` and hyper-parameter `cfg.param`

    - `convert_ymir_to_yolov5()` generate yolov5 dataset config file `data.yaml`

    - `write_ymir_training_result()` save model weight, map and other files.

    - `get_weight_file()` get pretrained weight or init weight file from ymir system

- modify `utils/datasets.py` for ymir dataset format

- modify `train.py` for training process monitor

- add `mining/data_augment.py` and `mining/mining_cald.py` for mining

- add `training/infer/mining-template.yaml` for `/img-man/training/infer/mining-template.yaml`

- add `cuda102/111.dockerfile`, remove origin `Dockerfile`

- modify `requirements.txt`

- other modify support onnx export, not important.

## new features

- 2022/09/08: add aldd active learning algorithm for mining task. [Active Learning for Deep Detection Neural Networks (ICCV 2019)](https://gitlab.com/haghdam/deep_active_learning)
- 2022/09/14: support change hyper-parameter `num_workers_per_gpu`
- 2022/09/16: support change activation, view [rknn](https://github.com/airockchip/rknn_model_zoo/tree/main/models/vision/object_detection/yolov5-pytorch)
- 2022/10/09: fix dist.destroy_process_group() hang
