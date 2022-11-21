# 镜像超参数

## ymir后台如何获取镜像超参数

- 通过解析镜像中 `/img-man/training-template.yaml` 获得训练的超参数， 若文件不存在则标记镜像不支持训练。

- 通过解析镜像中 `/img-man/infer-template.yaml` 获得推理的超参数，若文件不存在则标记镜像不支持推理。

- 通过解析镜像中 `/img-man/mining-template.yaml` 获得挖掘的超参数，若文件不存在则标记镜像不支持挖掘。

以 `youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi` 为例，可执行以下命令查看镜像对应超参数

```
docker run --rm youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi cat /img-man/training-template.yaml

# 输出结果
# training template for your executor app
# after build image, it should at /img-man/training-template.yaml
# key: gpu_id, task_id, pretrained_model_params, class_names should be preserved

# gpu_id: '0'
# task_id: 'default-training-task'
# pretrained_model_params: []
# class_names: []

shm_size: '128G'
export_format: 'ark:raw'
model: 'yolov5s'
batch_size_per_gpu: 16
num_workers_per_gpu: 4
epochs: 100
img_size: 640
opset: 11
args_options: '--exist-ok'
save_best_only: True  # save the best weight file only
save_period: 10
sync_bn: False  # work for multi-gpu only
ymir_saved_file_patterns: ''  # custom saved files, support python regular expression, use , to split multiple pattern
```

注：同名镜像在后台更新超参数配置文件如 `/img-man/training-template.yaml` 后，需要在 ymir 网页端重新添加，使超参数配置生效。

## 如何更新镜像默认的超参数

准备以下文件与对应内容:

- training-template.yaml

    ```
    model: 'yolov5n'  # change from yolov5s --> yolov5n
    batch_size_per_gpu: 2  # change from 16 --> 2
    num_workers_per_gpu: 2  # change from 4 --> 2
    epochs: 100
    img_size: 640
    opset: 12   # change from 11 --> 12
    args_options: '--exist-ok'
    save_best_only: True  # save the best weight file only
    save_period: 10
    sync_bn: False  # work for multi-gpu only
    ```

- zzz.dockerfile

```
FROM youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi

COPY ./training-template.yaml /img-man/training-template.yaml

CMD bash /usr/bin/start.sh
```

- 执行构建命令即可获得新镜像 `youdaoyzbx/ymir-executor:ymir2.0.1-yolov5-cu111-tmi`

```
docker build -t youdaoyzbx/ymir-executor:ymir2.0.1-yolov5-cu111-tmi . -f zzz.dockerfile
```

## 如何增加或删除镜像的超参数

准备以下文件与对应代码, 以修改镜像中 `/app/start.py` 为例

- training-template.yaml

- start.py: 修改该文件中的内容处理增加或删除的超参数

- zzz.dockerfile

```
FROM youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi

COPY ./training-template.yaml /img-man/training-template.yaml
COPY ./start.py /app/start.py

CMD bash /usr/bin/start.sh
```

- 执行构建命令即可获得新镜像 `youdaoyzbx/ymir-executor:ymir2.0.2-yolov5-cu111-tmi`

```
docker build -t youdaoyzbx/ymir-executor:ymir2.0.2-yolov5-cu111-tmi . -f zzz.dockerfile
```
