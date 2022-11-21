# a docker file for an sample training / mining / infer executor

FROM python:3.8.13-alpine

# Add bash
RUN apk add bash
# Required to build numpy wheel
RUN apk add g++

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

WORKDIR /app
# copy user code to WORKDIR
COPY ./app/start.py /app/

# copy user config template to /img-man
RUN mkdir -p /img-man
COPY img-man/*-template.yaml /img-man/

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
RUN echo "python /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
