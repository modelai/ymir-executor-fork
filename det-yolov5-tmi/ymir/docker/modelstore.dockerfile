# docker build -t youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu102-tmi -f fast.dockerfile .
FROM youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-modelstore
ARG YMIR=1.1.0
ENV YMIR_VERSION=${YMIR}
# fix font download directory
ENV YOLOV5_CONFIG_DIR='/root/.config/Ultralytics'

COPY . /app
RUN mv /app/*-template.yaml /img-man
RUN pip uninstall -y ymir_exc && pip install "git+https://github.com/yzbx/ymir-executor-sdk.git@ymir1.3.0"

RUN echo "python3 /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
