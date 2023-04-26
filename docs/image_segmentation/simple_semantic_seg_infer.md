# 制作一个简单的语义分割推理镜像

参考[ymir镜像制作简介](../overview/ymir-executor.md)

## 镜像输入输出示例
```
.
├── in
│   ├── annotations
│   ├── assets
│   ├── candidate-index.tsv
│   ├── config.yaml
│   ├── env.yaml
│   └── models
└── out
    ├── monitor.txt
    └── infer-result.json
```

## 工作目录

```
cd seg-semantic-demo-tmi
```

## 提供超参数模型文件

镜像中包含**/img-man/infer-template.yaml** 表示镜像支持推理

- [img-man/infer-template.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/img-man/infer-template.yaml)

```yaml
{!seg-semantic-demo-tmi/img-man/infer-template.yaml!}
```

- [Dockerfile](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/Dockerfile)

```
RUN mkdir -p /img-man  # 在镜像中生成/img-man目录
COPY img-man/*.yaml /img-man/  # 将主机中img-man目录下的所有yaml文件复制到镜像/img-man目录
```

## 提供镜像说明文件

**object_type** 为 3 表示镜像支持语义分割

- [img-man/manifest.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/img-man/manifest.yaml)
```
# 3 for semantic segmentation
"object_type": 3
```

- Dockerfile
`COPY img-man/*.yaml /img-man/` 在复制infer-template.yaml的同时，会将manifest.yaml复制到镜像中的**/img-man**目录

## 提供默认启动脚本

- Dockerfile
```
RUN echo "python /app/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

- [app/start.py](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-semantic-demo-tmi/app/start.py)

::: seg-semantic-demo-tmi.app.start._run_infer
    handler: python
    options:
      show_root_heading: false
      show_source: true

## 写进度

```
# use `monitor.write_monitor_logger` to write log to console and write task process percent to monitor.txt
logging.info(f"assets count: {len(lines)}, valid: {valid_image_count}")
monitor.write_monitor_logger(percent=0.2)

# real-time monitor
monitor.write_monitor_logger(percent=0.2 + 0.8 * iter / valid_image_count)

# if task done, write 100% percent log
logging.info('infer done')
monitor.write_monitor_logger(percent=1.0)
```

## 写结果文件

```
coco_results = convert(cfg, results, True)
rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')
```

- 结果文件格式

    - 参考 [coco-formats](https://cocodataset.org/#format-results)

    - 其中 RLE 为一种mask编码格式， 可通过pycocotools生成

```
{
    "categories": [{"id": int, "name": str, "supercategory": str}],
    "images": [{"id": int, "file_name": str, "width": int, "height" int}],
    "annotations": [{"id": int, "image_id": int, "category_id": int, "segmentation": RLE}]
}
```

- 结果文件示例
```
{
    "categories":[
        {
            "id":1,
            "name":"dog",
            "supercategory":"none"
        },
        {
            "id":2,
            "name":"cat",
            "supercategory":"none"
        }
    ],
    "images":[
        {
            "id":1,
            "file_name":"5ec2163001ed53f2169c525ff2e5e5ec.jpg",
            "width":1280,
            "height":854
        }
    ],
    "annotations":[
        {
            "id":5,
            "image_id":1,
            "category_id":1,
            "segmentation":{
                "size":[
                    854,
                    1280
                ],
                "counts":"iZ83cj00k_QQ1"
            }
        }
    ]
}
```

## 制作镜像 demo/semantic_seg:infer

```dockerfile
{!seg-semantic-demo-tmi/Dockerfile!}
```

```
docker build -t demo/semantic_seg:infer -f Dockerfile .
```
