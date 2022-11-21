# ymir-executor documentation [English](./README.MD) | [简体中文](./README_zh-CN.MD)

- [ymir](https://github.com/IndustryEssentials/ymir)

  - [bilibili: video tutorial](https://b23.tv/KS5b5oF)

- [wiki](https://github.com/modelai/ymir-executor-fork/wiki)

- [ymir executor](./docs/official-docker-image.md)

  - [develop tutorial](./docs/README.MD) from zero to one, build you ymir-executor

  - [user survey](https://www.wjx.cn/vm/eKFm2aq.aspx#) help ymir team to write document and develop, user feedback are collected on a continuous and long-term basis.

- [ymir mining algorithm](./docs/mining-images-overview.md)

## overview

| docker image | [finetune](https://github.com/modelai/ymir-executor-fork/wiki/use-yolov5-to-finetune-or-training-model) | tensorboard | args/cfg options | framework | onnx | pretrained weights |
| - | - | - | - | - | - | - |
| yolov4 | ? | ✔️ | ❌ | darknet + mxnet | ❌ | local |
| yolov5 | ✔️ | ✔️ | ✔️ | pytorch | ✔️ | local+online |
| yolov7 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | local+online |
| mmdetection | ✔️ | ✔️ | ✔️ | pytorch | ❌ | local+online |
| detectron2 | ✔️ | ✔️ | ✔️ | pytorch | ❌ | online |
| vidt | ? | ✔️ | ✔️ | pytorch | ❌ | online |
| nanodet | ✔️ | ✔️ | ❌ | pytorch_lightning | ❌ | local+online |

- `online` pretrained weights may download through network

- `local` pretrained weights have copied to docker images when building image

### benchmark

- training dataset: voc2012-train 5717 images
- validation dataset: voc2012-val 5823 images
- image size: 640

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

## how to import pretrained model weights

- [import and finetune model](https://github.com/modelai/ymir-executor-fork/wiki/import-and-finetune-model)

- [import pretainted model weights](https://github.com/IndustryEssentials/ymir/blob/master/dev_docs/import-extra-models.md)
## reference

- [yolov4](https://github.com/AlexeyAB/darknet)
- [yolov5](https://github.com/ultralytics/yolov5)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [yolov7](https://github.com/wongkinyiu/yolov7)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [vidt](https://github.com/naver-ai/vidt)
- [nanodet](https://github.com/RangiLyu/nanodet)
