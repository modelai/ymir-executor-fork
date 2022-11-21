# ymir-executor 使用文档 [English](./README.MD) | [简体中文](./README_zh-CN.MD)

- [ymir](https://github.com/IndustryEssentials/ymir)

  - [bilibili 视频教程](https://b23.tv/KS5b5oF)

- [说明文档](https://github.com/modelai/ymir-executor-fork/wiki)

- [ymir镜像](./docs/official-docker-image.md)

  - [镜像开发者：制作教程](./docs/README.MD) 从零到一，搭建自己的 ymir-executor

  - [用户调查](https://www.wjx.cn/vm/eKFm2aq.aspx#) 帮助ymir团队开发文档与新功能，此调查长期有效， 定期收集

- [ymir 挖掘算法](./docs/mining-images-overview.md)

## 比较

| docker image | [finetune](https://github.com/modelai/ymir-executor-fork/wiki/use-yolov5-to-finetune-or-training-model) | tensorboard | args/cfg options | framework | onnx | pretrained weight |
| - | - | - | - | - | - | - |
| yolov4 | ? | ✔️ | ❌ | darknet + mxnet | ❌ | local |
| yolov5 | ✔️ | ✔️ | ✔️ | pytorch | ✔️ | local+online |
| yolov7 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | local+online |
| mmdetection | ✔️ | ✔️ | ✔️ | pytorch | ❌ | local+online |
| detectron2 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | online |
| vidt | ? | ✔️ | ✔️ | pytorch | ❌ | online |
| nanodet | ✔️ | ✔️ | ❌ | pytorch_lightning | ❌ | local+online |

- `online` 预训练权重可能在训练时通过网络下载

- `local` 预训练权重在构建镜像时复制到了镜像

### benchmark

- 训练集: voc2012-train 5717 images
- 测试集: voc2012-val 5823 images
- 图像大小: 640 (nanodet为416, yolov4为608)

**由于 coco 数据集包含 voc 数据集中的类, 因此这个对比并不公平, 仅供参考**

gpu: single Tesla P4

| docker image | batch size | epoch number | model | voc2012 val map50 | training time | note |
| - | - | - | - | - | - | - |
| yolov5 | 16 | 100 | yolov5s | 70.05% | 9h | coco-pretrained |
| vidt | 2 | 100 | swin-nano | 54.13% | 2d | imagenet-pretrained |
| yolov4 | 4 | 20000 steps | yolov4 | 66.18% | 2d | imagenet-pretrained |
| yolov7 | 16 | 100 | yolov7-tiny | 70% | 8h | coco-pretrained |

gpu: single GeForce GTX 1080 Ti

| docker image | image size | batch size | epoch number | model | voc2012 val map50 | training time | note |
| - | - | - | - | - | - | - | - |
| yolov4 | 608 | 64/32 | 20000 steps | yolov4 | 72.73% | 6h | imagenet-pretrained |
| yolov5 | 640 | 16 | 100 | yolov5s | 70.35% | 2h | coco-pretrained |
| yolov7 | 640 | 16 | 100 | yolov7-tiny | 70.4% | 5h | coco-pretrained |
| mmdetection | 640 | 16 | 100 | yolox_tiny | 66.2% | 5h | coco-pretrained |
| detectron2 | 640 | 2 | 20000 steps | retinanet_R_50_FPN_1x | 53.54% | 2h | imagenet-pretrained |
| nanodet | 416 | 16 | 100 | nanodet-plus-m_416 | 58.63% | 5h | imagenet-pretrained |

---

## 如何导入预训练模型

- [如何导入并精调外部模型](https://github.com/modelai/ymir-executor-fork/wiki/import-and-finetune-model)

- [如何导入外部模型](https://github.com/IndustryEssentials/ymir/blob/master/dev_docs/import-extra-models.md)

    - 通过ymir网页端的 `模型管理/模型列表/导入模型` 同样可以导入模型

## 参考

- [yolov4](https://github.com/AlexeyAB/darknet)
- [yolov5](https://github.com/ultralytics/yolov5)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [yolov7](https://github.com/wongkinyiu/yolov7)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [vidt](https://github.com/naver-ai/vidt)
- [nanodet](https://github.com/RangiLyu/nanodet)
