# Please refer to the TRAINING documentation, "Basic Dockerfile for training"
FROM nvidia/cuda:9.0-base-ubuntu16.04

WORKDIR /noiseGAN

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        bash-completion \
        build-essential \
        libbz2-dev \
        locales \
        python3-venv \
        python3-pip \
        curl \
        unzip

# Prepare deps
RUN pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3

# Install libs Python dependencies
ADD requirements.txt /noiseGAN/
RUN pip3 install torch==0.3.1  torchvision==0.2.1
#https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl torchvision==0.2.1
RUN pip3 install -r requirements.txt

ADD . /noiseGAN
RUN bash ./download_trained_models.sh

RUN pip --version
RUN pip list 

CMD ["python3 generate_noise.py"]
