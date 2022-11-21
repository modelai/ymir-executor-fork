FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
ARG PIP_SOURCE=https://pypi.mirrors.ustc.edu.cn/simple

ENV PYTHONPATH=.

WORKDIR /darknet
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && apt-get update
RUN apt install -y software-properties-common wget
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y python3.7 python3-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN python3 get-pip.py
RUN pip3 install -i ${PIP_SOURCE} mxnet-cu112==1.9.1 numpy opencv-python pyyaml watchdog tensorboardX six scipy tqdm imagesize

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y libopencv-dev
COPY . /darknet
RUN make -j

RUN mkdir /img-man && cp /darknet/training-template.yaml /img-man/training-template.yaml && cp /darknet/mining/*-template.yaml /img-man
RUN echo "python3 /darknet/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
