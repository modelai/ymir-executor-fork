# FAQ

## 关于cuda版本

- 推荐主机安装高版本驱动，支持11.2以上的cuda版本, 使用11.1及以上的镜像

- GTX3080/GTX3090不支持11.1以下的cuda，只能使用cuda11.1及以上的镜像

## apt 或 pip 安装慢或出错

- 采用国内源，如在docker file 中添加如下命令

    ```
    RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

    RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
    ```

## docker build 的时候出错，找不到相应docker file或`COPY/ADD`时出错

- 回到项目根目录或docker file对应根目录，确保docker file 中`COPY/ADD`的文件与文件夹能够访问，以yolov5为例.

    ```
    cd ymir-executor/det-yolov5-tmi

    docker build -t ymir-executor/yolov5:cuda111 . -f cuda111.dockerfile
    ```

## 模型精度/速度如何权衡与提升

- 模型精度与数据集大小、数据集质量、学习率、batch size、 迭代次数、模型结构、数据增强方式、损失函数等相关，在此不做展开，详情参考：

    - [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)

    - [Paper with Code: Object Detection](https://paperswithcode.com/task/object-detection)

    - [awesome object detection](https://github.com/amusi/awesome-object-detection)

    - [voc2012 object detection leadboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

    - [coco object detection leadboard](https://cocodataset.org/#detection-leaderboard)
