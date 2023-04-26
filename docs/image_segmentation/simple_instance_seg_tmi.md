# 制作简单的实例分割镜像

参考语义分割镜像的制作:

- [语义分割-训练](./simple_semantic_seg_training.md)

- [语义分割-推理](./simple_semantic_seg_infer.md)

- [语义分割-挖掘](./simple_semantic_seg_mining.md)

## 镜像说明文件

**object_type** 为 4 表示镜像支持实例分割

- [img-man/manifest.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-instance-demo-tmi/img-man/manifest.yaml)
```
# 4 for instance segmentation
"object_type": 4
```

## 训练结果返回

```
rw.write_model_stage(stage_name='epoch20',
                     files=['epoch20.pt', 'config.py'],
                     evaluation_result=dict(maskAP=expected_maskap))
```

## 推理结果返回

采用coco数据集格式，相比语义分割，实例分割的annotation中需要增加 `bbox` 的置信度。
```
# for instance segmentation
annotation_info['confidence'] = min(1.0, 0.1 + random.random())

coco_results = convert(cfg, results, True)
rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')
```

- 结果文件格式

    - 参考 [coco-formats](https://cocodataset.org/#format-results)

    - 其中 RLE 为一种mask编码格式， 可通过pycocotools生成

    - 其中 bbox 格式为 xywh
```
{
    "categories": [{"id": int, "name": str, "supercategory": str}],
    "images": [{"id": int, "file_name": str, "width": int, "height" int}],
    "annotations": [{"id": int, "image_id": int, "category_id": int, "segmentation": RLE, "bbox": [x1, y1, w, h], "confidence": float}]
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
            "confidence":0.9,
            "bbox":[
                9,
                851,
                2,
                3
            ],
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
