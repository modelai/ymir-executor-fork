ARG baseimage=21.02-py3
# 21.02-py3: ubuntu20.04, cuda11.2.0, pytorch 1.8.0a0+52ea372, TensorRT 7.2.2.3+cuda11.1.0.024
# 20.12-py3: pytorch version 1.8.0a0+1606899 not support onnx silu/hardswish export
# yolov5 recommended for 21.08-py3, here we use py3.8 + cuda11.1.1 + pytorch1.8.0
# view https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12 for details
FROM nvcr.io/nvidia/pytorch:${baseimage}
ARG YMIR="1.1.0"
ENV PYTHONPATH=.
ENV YMIR_VERSION=$YMIR
# fix font download directory
ENV YOLOV5_CONFIG_DIR='/app/data'

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY . /app
# change apt and pypy mirrors
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && apt update \
    && apt install -y zip htop vim libgl1-mesa-glx libopencv-dev \
    && apt install -y tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# install ymir-exc and object_detection_metrics
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.3.0" && \
    pip install "git+https://github.com/modelai/review_object_detection_metrics@ymir" && \
    pip install -r /app/ymir/tensorrt-requirements.txt && \
    mkdir -p /img-man && \
    mv /app/ymir/img-man/*.yaml /img-man && \
    mkdir -p ${YOLOV5_CONFIG_DIR} && \
    mv /app/ymir/*.ttf ${YOLOV5_CONFIG_DIR} && \
    mv /app/ymir/*.pt /app && \
    echo "cd /app && python3 ymir/start.py && python3 ymir/tensorrt/ymir_tensorrt.py" > /usr/bin/start.sh

WORKDIR /app

# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
