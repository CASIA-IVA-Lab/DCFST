#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing python and numpy ******************"
conda install -y python=3.7.3
conda install -y numpy=1.16.3
conda install -y numpy-base=1.16.3

echo ""
echo ""
echo "****************** Installing pytorch 0.4.1 with cuda80 ******************"
conda install -y pytorch=0.4.1 torchvision=0.2.1 cuda80 -c pytorch 

echo ""
echo ""
echo "****************** Installing matplotlib 3.1.3 ******************"
conda install -y matplotlib=3.1.3

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas=1.0.3

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python==4.0.1.24

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboard
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing nltk ******************"
pip install nltk==3.4.1

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython=0.29.15

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py


echo ""
echo ""
echo "****************** Installing Shapely for VOT challenge ******************"
pip install Shapely==1.6.4.post2

echo ""
echo ""
echo "****************** Compile RoI ******************"
cd ltr/external/PreciseRoIPooling/pytorch/prroi_pool
bash travis.sh
cd ../../../../
