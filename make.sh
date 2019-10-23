#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/opt/cuda/
CUDAHOME=/opt/cuda/

python3 build.py build_ext --inplace

cd ..
