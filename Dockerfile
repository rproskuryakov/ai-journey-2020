FROM nvidia/cuda:11.1-base-ubuntu18.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip git -y

COPY ./requirements.txt .
RUN pip3 install torch==1.6.0 torchvision==0.7.0 pytorch-nlp==0.5.0 && \
    pip3 install -r requirements.txt