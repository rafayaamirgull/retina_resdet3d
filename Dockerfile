#####
## Author: RAFAYAAMIR
## Date:   Sep 25 2024
#####

# BASE IMAGE HAVING CUDA 11.3.1 AND UBUNTU 20.04
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS x86_64_build

ENV DEBIAN_FRONTEND=noninteractive
ENV BASE_DIR /workspace
WORKDIR ${BASE_DIR}


RUN apt update &&\
    apt install -y \
    software-properties-common \
    build-essential \
    gnupg2 \
    wget \
    unzip

# DOWNLOAD AND INSTALL PYTHON AND PIP
RUN apt update &&\
    apt install -y \
    curl \
    python3-pip

RUN pip3 --default-timeout=1000 install torchvision tensorboardx
RUN pip3 install scipy==1.10.1 pyyaml==6.0.2 easydict==1.13

COPY ./ /workspace
ENV PYTHONPATH="/workspace"
ENV PYTHONIOENCODING=UTF-8
