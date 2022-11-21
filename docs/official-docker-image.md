# official docker image

update: 2022/11/01

## the hyper-parameters for ymir-executor

| docker images | epochs/iters | model structure | image size | batch_size |
| - | - | - | - | - |
| yolov5 | epochs | model | img_size | batch_size_per_gpu |
| mmdetection | max_epochs | config_file | - | samples_per_gpu |
| yolov4 | max_batches | - | image_height, image_width | batch |
| yolov7 | epochs | cfg_file | img_size | batch_size_per_gpu |
| nanodet | epochs | config_file | input_size | batch_size_per_gpu |
| vidt | epochs | backbone_name | eval_size | batch_size_per_gpu |
| detectron2 | max_iter | config_file | - | batch_size |

- epochs: such as `epochs` or `max_epochs`, control the time for training.
- iters: such as `max_batches` or `max_iter`, control the time for training.
- ymir_saved_file_patterns: save the file match one of the pattern. for example `best.pt, *.yaml` will save `best.pt` and all the `*.yaml` file in `/out/models` directory.
- export_format: the dataset format for ymir-executor in `/in`, support `ark:raw` and `voc:raw`
- args_options/cfg_options: for yolov5, use it for other options, such as `--multi-scale --single-cls --optimizer SGD` and so on, view `train.py, parse_opt()` for detail. for mmdetection and detectron2, it provides methods to change other hyper-pameters not defined in `/img-man/training-template.yaml`

## docker image format

youdaoyzbx/ymir-executor:[ymir-version]-[repository]-[cuda version]-[ymir-executor function]

- ymir-version
    - ymir1.1.0
    - ymir1.2.0
    - ymir1.3.0
    - ymir2.0.0

- repository
    - yolov4
    - yolov5
    - yolov7
    - mmdet
    - detectron2
    - vidt
    - nanodet

- cuda version
    - cu101: cuda 10.1
    - cu102: cuda 10.2
    - cu111: cuda 11.1
    - cu112: cuda 11.2

- ymir-executor function
    - t: training
    - m: mining
    - i: infer
    - d: deploy



## ymir2.0.0

2022/10/26: support ymir1.1.0/1.2.0/1.3.0/2.0.0

```
youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-yolov7-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-mmdet-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-detectron2-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-vidt-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-nanodet-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmid # support deploy
youdaoyzbx/ymir-executor:ymir2.0.0-yolov4-cu111-tmi  # deprecated
```

## ymir1.3.0

2022/10/10: support ymir1.1.0/1.2.0/1.3.0/2.0.0

```
youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-tmi
youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-v6.2-cu111-tmi
youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-modelstore
youdaoyzbx/ymir-executor:ymir1.3.0-mmdet-cu111-tmi
```

## ymir1.1.0

- [yolov4](https://github.com/modelai/ymir-executor-fork#det-yolov4-training)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu112-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov4-cu101-tmi
    ```

- [yolov5](https://github.com/modelai/ymir-executor-fork#det-yolov5-tmi)

    - [change log](./det-yolov5-tmi/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu102-tmi
    ```

- [mmdetection](https://github.com/modelai/ymir-executor-fork#det-mmdetection-tmi)

    - [change log](./det-mmdetection-tmi/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu111-tmi

    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-mmdet-cu102-tmi
    ```

- [detectron2](https://github.com/modelai/ymir-detectron2)

    - [change log](https://github.com/modelai/ymir-detectron2/blob/master/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-detectron2-cu111-tmi
    ```

- [yolov7](https://github.com/modelai/ymir-yolov7)

    - [change log](https://github.com/modelai/ymir-yolov7/blob/main/ymir/README.md)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov7-cu111-tmi
    ```

- [vidt](https://github.com/modelai/ymir-vidt)

    - [change log](https://github.com/modelai/ymir-vidt/tree/main/ymir)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-vidt-cu111-tmi
    ```

- [nanodet](https://github.com/modelai/ymir-nanodet/tree/ymir-dev)

    - [change log](https://github.com/modelai/ymir-nanodet/tree/ymir-dev/ymir)

    ```
    docker pull youdaoyzbx/ymir-executor:ymir1.1.0-nanodet-cu111-tmi
    ```

# build ymir executor

## det-yolov4-tmi

- yolov4 training, mining and infer docker image, use `mxnet` and `darknet` framework

  ```
  cd det-yolov4-tmi
  docker build -t ymir-executor/yolov4:cuda101-tmi -f cuda101.dockerfile .

  docker build -t ymir-executor/yolov4:cuda112-tmi -f cuda112.dockerfile .
  ```

## det-yolov5-tmi

- yolov5 training, mining and infer docker image, use `pytorch` framework

```
cd det-yolov5-tmi
docker build -t ymir-executor/yolov5:cuda102-tmi -f cuda102.dockerfile .

docker build -t ymir-executor/yolov5:cuda111-tmi -f cuda111.dockerfile .
```

## det-mmdetection-tmi

```
cd det-mmdetection-tmi
docker build -t ymir-executor/mmdet:cu102-tmi -f docker/Dockerfile.cuda102 .

docker build -t ymir-executor/mmdet:cu111-tmi -f docker/Dockerfile.cuda111 .
```
