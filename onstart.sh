#!/bin/bash
apt-get update
apt-get --yes install curl
apt-get --yes install git
conda install -y -c conda-forge faiss
conda install -y -c conda-forge mkl
!nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free