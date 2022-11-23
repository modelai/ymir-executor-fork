FROM youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-tensorrt-cu111-tmi

COPY . /app
RUN echo "cd /app && python3 ymir/tensorrt/ymir_tensorrt.py" > /usr/bin/start.sh

WORKDIR /app

# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
