# yolov5-ymir readme
- [yolov5 readme](./README_yolov5.md)

```
docker build -t ymir/ymir-executor:ymir1.1.0-cuda102-yolov5-tmi --build-arg SERVER_MODE=dev --build-arg YMIR=1.1.0 -f cuda102.dockerfile .

docker build -t ymir/ymir-executor:ymir1.1.0-cuda111-yolov5-tmi --build-arg SERVER_MODE=dev --build-arg YMIR=1.1.0 -f cuda111.dockerfile .
```

## main change log

- add `start.py` and `utils/ymir_yolov5.py` for train/infer/mining

- add `utils/ymir_yolov5.py` for useful functions

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
- 2022/09/28: support rknn model quantification
