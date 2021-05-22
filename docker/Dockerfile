FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && \
    apt-get install -y sudo curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev \
    protobuf-compiler apt-utils nano vim man build-essential wget && \
    curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y python3.6

RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && python3.6 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy scipy pyyaml matplotlib imageio pygame imageio-ffmpeg tensorboard-logger ruamel.base ryd jsonpickle==0.9.6

RUN mkdir /install
WORKDIR /install

# install Sacred
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3.6 setup.py install

# Install pymongo
RUN pip3 install pymongo

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch==1.1.0

## -- SMAC
ENV smac_ver 1
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /REFIL/3rdparty/StarCraftII

WORKDIR /REFIL
