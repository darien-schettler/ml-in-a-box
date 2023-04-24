#!/bin/bash

# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Set ENV variables
    export APT_INSTALL="apt-get install -y --no-install-recommends"
    export PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
    export GIT_CLONE="git clone --depth 10"

    # Update apt
    sudo apt update


# ==================================================================
# Tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev


# ==================================================================
# Python
# ------------------------------------------------------------------

    #Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.9
    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y

    # Installing python3.9
    DEBIAN_FRONTEND=noninteractive sudo $APT_INSTALL \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-distutils-extra

    # Add symlink so python and python3 commands use same python3.9 executable
    sudo ln -s /usr/bin/python3.9 /usr/local/bin/python3
    sudo ln -s /usr/bin/python3.9 /usr/local/bin/python

    # Installing pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
    export PATH=$PATH:/home/paperspace/.local/bin
    
    # Upgrade pip
    python -m pip install --upgrade pip

# ==================================================================
# Installing CUDA packages (CUDA Toolkit 11.8 & CUDNN 8.6)
# ------------------------------------------------------------------

    # Install CUDA Toolkit 11.8.0
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
    export PATH=$PATH:/usr/local/cuda-11.8/bin
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
    rm cuda_11.8.0_520.61.05_linux.run

    # Install libcudnn8 and libcudnn8-dev
    sudo $APT_INSTALL libcudnn8=8.6.0.*-1+cuda11.8
    sudo $APT_INSTALL libcudnn8-dev=8.6.0.*-1+cuda11.8

# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/
    $PIP_INSTALL torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda
    $PIP_INSTALL --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ==================================================================
# TensorFlow
# ------------------------------------------------------------------
    # Install keras-nlp
    $PIP_INSTALL keras-nlp

    # Install TensorFlow
    $PIP_INSTALL tensorflow

# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Install Hugging Face transformers and datasets
    $PIP_INSTALL transformers datasets


# ==================================================================
# Jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL jupyterlab==3.4.6

# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image \
        scikit-learn \
        matplotlib \
        ipython \
        ipykernel \
        ipywidgets \
        cython \
        tqdm \
        gdown \
        xgboost \
        pillow \
        seaborn \
        sqlalchemy \
        spacy \
        nltk \
        boto3 \
        tabulate \
        future \
        gradient \
        jsonify \
        opencv-python \
        pyyaml \
        sentence-transformers \
        wandb

# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

    sudo $APT_INSTALL default-jre
    sudo $APT_INSTALL default-jdk


# ==================================================================
# CMake
# ------------------------------------------------------------------

    sudo $GIT_CLONE https://github.com/Kitware/CMake ~/cmake
    cd ~/cmake
    sudo ./bootstrap
    sudo make -j"$(nproc)" install


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    sudo curl -sL https://deb.nodesource.com/setup_16.x | sudo bash
    sudo $APT_INSTALL nodejs
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git
    DEBIAN_FRONTEND=noninteractive jupyter contrib nbextension install --user

# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    echo "export PATH=${PATH}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc

