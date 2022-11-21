# det-mmdetection-tmi

- [mmdetection](./README_mmdet.md)

`mmdetection` framework for object `det`ection `t`raining/`m`ining/`i`nfer task

# build docker image

```
docker build -t ymir-executor/mmdet:cuda102-tmi --build-arg YMIR=1.1.0 -f docker/Dockerfile.cuda102 .

docker build -t ymir-executor/mmdet:cuda111-tmi --build-arg YMIR=1.1.0 -f docker/Dockerfile.cuda111 .
```


# changelog
- modify `mmdet/datasets/coco.py`, save the evaluation result to `os.environ.get('COCO_EVAL_TMP_FILE')` with json format
- modify `mmdet/core/evaluation/eval_hooks.py`, write training result file and monitor task process
- modify `mmdet/datasets/__init__.py, mmdet/datasets/coco.py` and add `mmdet/datasets/ymir.py`, add class `YmirDataset` to load YMIR dataset.
- modify `requirements/runtime.txt` to add new dependent package.
- add `mmdet/utils/util_ymir.py` for ymir training/infer/mining
- add `ymir_infer.py` for infer
- add `ymir_mining.py` for mining
- add `ymir_train.py` modify `tools/train.py` to update the mmcv config for training
- add `start.py`, the entrypoint for docker image
- add `training-template.yaml, infer-template.yaml, mining-template.yaml` for ymir pre-defined hyper-parameters.
- add `docker/Dockerfile.cuda102, docker/Dockerfile.cuda111` to build docker image
- remove `docker/Dockerfile` to avoid misuse

---

- 2022/09/06: set `find_unused_parameters = True`, fix DDP bug
- 2022/10/18: add `random` and `aldd` mining algorithm. `aldd` algorithm support yolox only.
- 2022/10/19: fix training class_number bug in `recursive_modify_attribute()`
