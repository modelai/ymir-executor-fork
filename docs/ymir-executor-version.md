# ymir1.3.0 (2022-09-30)

- 支持分开输出模型权重，用户可以采用epoch10.pth进行推理，也可以选择epoch20.pth进行推理

- 训练镜像需要指定数据集标注格式, ymir1.1.0默认标注格式为`ark:raw`, ymir1.3.0默认标注格式为`ark:voc`

- 训练镜像可以从`/in/env.yaml`中获得系统ymir的接口版本`protocol_version`，方便镜像做兼容
    - 对于ymir1.3.0, 定义protocol_version为 1.0.0
```
task_id: t000000100000166d7761660213748
protocol_version: 1.0.0
run_training: true
run_mining: false
run_infer: false
input:
  annotations_dir: /in/annotations
  assets_dir: /in/assets
  candidate_index_file: /in/candidate-index.tsv
  config_file: /in/config.yaml
  models_dir: /in/models
  root_dir: /in
  training_index_file: /in/train-index.tsv
  val_index_file: /in/val-index.tsv
output:
  infer_result_file: /out/infer-result.json
  mining_result_file: /out/result.tsv
  models_dir: /out/models
  monitor_file: /out/monitor.txt
  root_dir: /out
  tensorboard_dir: /out/tensorboard
  training_result_file: /out/models/result.yaml
```

## 辅助库

- [ymir-executor-sdk](https://github.com/modelai/ymir-executor-sdk) 采用ymir1.3.0分支

- [ymir-executor-verifier](https://github.com/modelai/ymir-executor-verifier) 镜像检查工具，从ymir1.3.0开始支持

# ymir1.1.0

- [custom ymir-executor](https://github.com/IndustryEssentials/ymir/blob/dev/dev_docs/ymir-dataset-zh-CN.md)

- [ymir-executor-sdk](https://github.com/modelai/ymir-executor-sdk) 采用ymir1.0.0分支
