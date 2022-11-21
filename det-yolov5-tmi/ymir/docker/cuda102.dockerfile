ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="1.1.0"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8
ENV YMIR_VERSION=${YMIR}
ENV YOLOV5_CONFIG_DIR='/app/data'

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install ymir-exc sdk
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.3.0"

# Copy file from host to docker and install requirements
COPY . /app
RUN mkdir /img-man && mv /app/ymir/img-man/*-template.yaml /img-man/ \
    && pip install -r /app/requirements.txt

# Download pretrained weight and font file
RUN cd /app && bash data/scripts/download_weights.sh \
    && wget https://ultralytics.com/assets/Arial.ttf -O ${YOLOV5_CONFIG_DIR}/Arial.ttf

# Make PYTHONPATH find local package
ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 /app/ymir/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
