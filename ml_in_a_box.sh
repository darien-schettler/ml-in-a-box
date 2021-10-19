# Install software for Ubuntu 20.04 ML-in-a-Box template VM on Paperspace Core
#
# Uses VM supplied by engineering
# After running, VM is templatized by them
# Script requires sudo access
# Installs go to standard locations, not ~/src as in previous 18.04 ML-in-a-Box template from 2018
# (except for Miniconda, required for NVidia RAPIDS)
# Script assumes a fresh copy of the VM, with the software to be installed not already present
# Uses a mix of install methods: Debian repo, apt-get, pip, Miniconda, depending on software
# Some pip installs have alternative apt-get installs, but pip versions for Python/ML tend to be newer
# See under the individual software installs below for other detail comments
#
# See also
#
# Paperspace-internal GitHub repo for this ML-in-a-Box setup: https://github.com/Paperspace/ml-in-a-box
#
# Last updated: Oct 04th 2021

### TODO ###

# These should be done before the VM is released

# Engineering QA the GPU/CUDA/NVidia setup, especially w.r.t.
#  - A100 GPU & CUDA so "it just works"
#  - Versioning (most things here are fixed version, some are not; ones not have no obvious way to specify a version)
#  - Security (we're using installed repos with arbitrary code)
#  - Run script in one go as the superuser? (commands require sudo; VM doesn't let me be SU)
#  - Licensing OK (looks good to me but I am not a lawyer)
#  - Test out final VM when built

# Improvements

# These improve the script but are not vital

# Add PyTorch 1.9/1.10 when CUDA 11.4 is supported
# CUDA toolkit install is not verified. Later steps use it but there is a verify install using manual steps.
# Put NumPy back to 1.21.2 after TensorFlow 2.5.0 downgrades it: pip install numpy 1.21.2 again, and then check TensorFlow still works
# Unclear what the lib errors are when TF is asked to show the GPU devices after install
# NVidia RAPIDS requires own virtualenv (conda, package installs, etc.), which is not ideal when nothing else installed needs it

# Working directory for script

mkdir ~/src
cd ~/src


# GPU
# ---

# - NVidia Driver 470.63.01 -

# VM is set up by engineering with latest driver as of Sep 15th 2021: 470.63.01

# - CUDA 11.4 -

# CUDA 11.4 is present on the VM as supplied by engineering

# - cuDNN 8.2.4.15-1+cuda11.4 -

# Following https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html section 2.3.4.1. Ubuntu Network Installation
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8=8.2.4.*-1+cuda11.4 -y
sudo apt-get install libcudnn8-dev=8.2.4.*-1+cuda11.4 -y

# CUDA toolkit

# Gives /usr/bin/nvcc, relevant to cuDNN
sudo apt-get install nvidia-cuda-toolkit=10.1.243-3 -y # Version was derived from --dry-run

# cuDNN install is not verified

# https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify says use /usr/src/cudnn_samples_v8/mnistCUDNN
# but the directory only contains NVIDIA_SLA_cuDNN_Support.txt, the license agreement
# It looks like the network install (2.3.4.1. Ubuntu Network Installation) doesn't have the code samples step corresponding to
# sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_{amd,arm}64.deb in the Debian install (2.3.2. Debian Installation)
# But that installation requires downloading cuDNN for Linux manually via the Nvidia website, not a script


# Infra
# -----

# Docker and NVidia-Docker so users can run their own containers, including with GPUs

# - Docker Engine -

# https://docs.docker.com/engine/install/ubuntu/
# Docker engine, X86_64 architecture
# Use the repo method
# There is a convenience script, but it doesn't refer to a specific version

# Commands for repo
sudo apt-get update
sudo apt-get install apt-transport-https=2.0.6 ca-certificates=20210119~20.04.2 -y # gnupg and lsb-release also needed, but already at latest versions

# Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up repo (stable, not nightly/test)
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce=5:20.10.8~3-0~ubuntu-focal docker-ce-cli=5:20.10.8~3-0~ubuntu-focal containerd.io -y

# Verify the install
sudo docker run hello-world


# - NVidia Docker 2.6 -

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Stable repo and GPG key

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Add experimental for A100 GPU features
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

# Install
sudo apt-get update
sudo apt-get install nvidia-docker2=2.6.0-1 -y
sudo systemctl restart docker

# Verify install
# Can see version with sudo nvidia-docker version, which shows more information than sudo nvidia-docker --version
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi


# Python
# ------

# Basic Python setup for Jupter-Notebook-based data science
# User can add their own software for their particular specialty

# - Python 3.8.10 -

# This is now installed from one of the above steps: run with "python3", not "python"

# - pip3 20.0.2 -

# Works as "pip" or "pip3"
# pip often has newer versions of ML packages than Ubuntu's apt

sudo apt-get update
sudo apt-get install python3-pip=20.0.2-5ubuntu1.6 -y

# - Numpy 1.21.2 -

pip3 install numpy==1.21.2

# - Pandas 1.3.3 -

pip3 install pandas==1.3.3

# - Matplotlib 3.4.3 -

pip3 install matplotlib==3.4.3

# - JupyterLab 3.1.12 -

# https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html
# JupyterLab includes Jupyter notebook

pip3 install jupyterlab==3.1.12

# Add /home/paperspace/.local/bin to path (numpy and jupyterlab warn on this)
export PATH=${PATH}:/home/paperspace/.local/bin

# Invoke with jupyter lab (not jupyterlab); appears on browser localhost:8888, or others


# ML
# --

# TensorFlow and PyTorch for deep learning
# H2O for other ML algorithms

# - H2O-3 3.34.0.1 -

# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html

# Dependencies

# Says pip install requests, tabulate, future; requests is present
# Java is already installed from earlier in this script

pip3 install tabulate==0.8.9
pip3 install future==0.18.2

# Install

pip3 install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o==3.34.0.1

# Verify install

# This can be done interactively in Python, using
# python3 -c "import h2o; h2o.init(); h2o.demo('glm')"


# - Scikit-learn 0.24.2 -

# https://scikit-learn.org/stable/install.html
# This also installs SciPy 1.7.1

pip3 install scikit-learn==0.24.2

# Verify install
python3 -c "import sklearn; sklearn.show_versions()"


# - TensorFlow 2.5.0 -

# https://www.tensorflow.org/install/pip

# GPU support is included by default; tensorflow-gpu is not required separately
# Note TF 2.5 not 2.6 yet

# Downgrades numpy to 1.19.5!

#  Attempting uninstall: numpy
#    Found existing installation: numpy 1.21.2
#    Uninstalling numpy-1.21.2:
#      Successfully uninstalled numpy-1.21.2

# May work better to exit and reenter the shell for TF to see the GPU, but it is found under devices

pip3 install tensorflow==2.5.0

# Verify install
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" # GPUs (https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d)


# - Nvidia RAPIDS 21.08 -

# Requires conda (Anaconda/Miniconda) install or Docker container
# Source builds are for contributors; non-dev is still conda
# Pip is not supported
# Appears to create own env with a bunch of installs, including CUDA Toolkit, but OK if it works

# https://rapids.ai/start.html
# https://docs.conda.io/en/latest/miniconda.html
# https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
# https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent (MacOS or Linux)

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh # Latest is Python 3.9, so get 3.8
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/src/miniconda # -b flag is silent mode install, -p is install prefix. sudo /root/miniconda3/bin/conda init below doesn't work without -p; with -p doesn't need sudo.
export PATH=${PATH}:$HOME/src/miniconda/bin
conda create -y -n rapids-21.08 -c rapidsai -c nvidia -c conda-forge rapids-blazing=21.08 python=3.8 cudatoolkit=11.4 # NVidia site says cudatoolkit 11.2 but runs with 11.4
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh

# Can briefly verify with the following (requires interactive steps)

# conda init bash
# (Then exit shell and reenter)
# conda activate rapids-21.08
# python3 -c "import cudf, cuml; print(cudf.__version__); print(cuml.__version__)"
# conda deactivate


# - PyTorch 1.9.0 -

# https://pytorch.org/get-started/locally/
# says pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# but this is CUDA 11.1

# https://discuss.pytorch.org/t/pytorch-with-cuda-11-compatibility/89254/15
# says pip build is CUDA 11.1, nightlies 11.3, 11.4 needs source build until PyTorch 1.10

# https://github.com/pytorch/pytorch#from-source
# Source build instructions say use conda install with magma-cuda from https://anaconda.org/pytorch/repo

# https://anaconda.org/pytorch/repo
# But magma-cuda only goes to magma-cuda113, which is CUDA 11.3 
