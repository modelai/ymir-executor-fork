# ymir镜像简介

ymir镜像为ymir平台提供数据训练，推理及挖掘功能，用户可以根据自己的需求下载使用不同的ymir镜像，也可以自行制作ymir镜像。

- 从数据的角度看，ymir平台实现了数据的导入、划分、合并与标注等功能；镜像则提供代码与环境依赖，利用数据训练模型，对数据进行推理或挖掘出最有标注价值的数据。

- 从镜像的角度看，ymir平台提供数据集、任务与超参数信息，镜像处理后产生结果文件，ymir对结果文件进行解析，并显示在ymir平台上。

- 从接口的角度看，约定好ymir平台提供的数据与超参数格式，镜像产生的结果文件格式。则可以提供多种镜像，实现不同的算法功能并对接到ymir平台。

!!! 注意
    与其它docker镜像不同，ymir镜像中包含镜像配置文件、代码与运行环境。

## ymir镜像使用

- [模型训练](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)

- [模型推理](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

- [数据挖掘](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E#%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98)

## ymir镜像

> 将ymir镜像视为一个对象或黑盒，它有以下属性

- 镜像类型：按镜像提供的功能，可以将镜像分类为训练镜像，推理镜像及挖掘镜像。一个镜像可以同时为训练，推理及挖掘镜像，也可以仅支持一种或两种功能。

    - ymir平台基于镜像或数据集，可以发起训练，推理及挖掘任务，任务信息提供到选择的镜像，启动对应的代码实现对应功能。如发起训练任务，将启动镜像中对应的训练代码；发起推理任务，将启动镜像中对应的推理代码。目前ymir平台支持发起单一任务，也支持发起推理及挖掘的联合任务。

- 镜像地址：来自[docker](https://www.runoob.com/docker/docker-tutorial.html)的概念，即镜像的仓库源加标签，一般采用<仓库源>:<标签>的格式，如 `ubuntu:22.04`, `youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi`。
    - 对于公开的镜像，仓库源对应docker hub上的镜像仓库，如 [youdaoyzbx/ymir-executor](https://hub.docker.com/r/youdaoyzbx/ymir-executor/tags), [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags)

- 镜像名称：用户自定义的镜像名称，注意名称长度，最多50个字符

- 镜像功能参数：为提高镜像的灵活性，用户可以在ymir平台上修改镜像的默认功能参数。如 `epochs`, `batch_size_per_gpu`，控制训练镜像的训练时长及显存占用。注意ymir平台为所有镜像提供额外的[通用参数](./hyper-parameter.md)

    - 训练镜像功能参数：对应训练超参数，常见的有`epochs`, `batch_size_per_gpu`, `num_workers_per_gpu`。默认训练参数配置文件存放在镜像的`/img-man/training-template.yaml`

    - 推理镜像功能参数：常见的有`confidence_threshold`，设置推理置信度。默认推理参数配置文件存放在镜像的`/img-man/infer-template.yaml`

    - 挖掘镜像功能参数：常见的有`confidence_threshold`设置推理置信度, `mining_algorithm`设置挖掘算法。默认挖掘参数配置文件存放在镜像的`/img-man/mining-template.yaml`

- 镜像目标：根据镜像中算法的类型，将镜像分为目标检测镜像、语义分割镜像及实例分割镜像等。

    - 镜像目标定义在镜像的 `/img-man/manifest.yaml` 文件中，如此文件不存在，ymir则默认镜像为目标检测镜像。

- 关联镜像：对于单一功能的镜像，训练镜像产生的模型，其它镜像不一定能使用。如采用基于[yolov4](https://github.com/AlexeyAB/darknet)训练的模型权重，基于[yolov7](https://github.com/WongKinYiu/yolov7) 推理镜像不支持加载相应模型权重。 因此需要对此类镜像进行关联，推荐使用多功能镜像。

!!! 添加镜像
    添加镜像时需要管理员权限，ymir平台首先会通过 `docker pull` 下载镜像，再解析镜像的`/img-man`目录，确定镜像中算法的类型及镜像支持的功能。


## ymir平台与镜像之间的接口

> 从镜像的角度看，ymir平台将任务信息，数据集信息，超参数信息放在镜像的`/in`目录，而镜像输出的进度信息，结果文件放在镜像的`/out`目录。

- 任务信息：任务信息包含是否要执行的训练，推理或挖掘任务，任务id。参考镜像文件[/in/env.yaml](../sample_files/in_env.md)

- [数据集信息](./dataset-format.md)：ymir平台中所有的数据集存放在相同的目录下，其中图片以其hash码命名，以避免图片的重复。ymir平台为镜像提供索引文件，索引文件的每一行包含图像绝对路径及对应标注绝对路径。

    - 对于训练任务，标注的格式由超参数 [export-format](./hyper-parameter.md) 决定。

    - 对于推理及挖掘任务，索引文件仅包含图像绝对路径。

    - 参考镜像文件 [/in/env.yaml](../sample_files/in_config.md)

- [超参数信息](./hyper-parameter.md)

- [接口文档](../design_doc/ymir_call_image.md)
