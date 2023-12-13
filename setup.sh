#!/bin/bash

#chmod +x pretrain.sh
#chmod +x finetune.sh
#chmod +x create_pretrain_csv.sh

######## GET DEPENDENCIES #########

## need this to run pretrain.sh (you'll need to click yes)
sudo apt-get install libgl1-mesa-glx
sudo apt-get install unzip

requirements_file="requirements.txt"

# Check if the requirements file exists
if [ ! -f "$requirements_file" ]; then
  echo "Error: File '$requirements_file' not found."
  exit 1
fi

 Read the requirements file line by line and install each module using pip
while IFS= read -r module; do
  if [[ "$module" != \#* ]]; then
    # Skip lines starting with '#' (comments) in the requirements file
    pip install "$module"
  fi
done < "$requirements_file"

echo "All modules from '$requirements_file' have been installed."



##### ASK WHICH TYPE OF ENVIRONMENT #####

# by defualt use pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel on vast.ai
# eva_clip requires:
# videoCLIP requires:

echo "which things to download"

read -p "default image teacher (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading default image teacher"
    curl https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth --output image_teacher.pth
fi

read -p "default video teacher (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading default video teacher"
    gdown 'https://drive.google.com/u/0/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=63e3a20f-2e32-4603-bc52-13a154ead88c&amp;at=ALt4Tm0vJDg8yrew90Qs81X3co6l:1691104865099&confirm=t&uuid=319d04ee-2975-41b8-b28e-2118e9b41167&at=ALt4Tm03Kkqy082aSFi3NPa54Un3:1691106288399' --output video_teacher.pth
fi

read -p "CLIP (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading CLIP"
    gdown https://drive.google.com/uc?id=1x9svwSPTXHVe21mF--q4IbMC-yGFOXFR --output clip_model.pth
fi

read -p "EVA-CLIP (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading EVA-CLIP"
    gdown https://drive.google.com/uc?id=1YQJ21--myZCs9hKwpRsetqW4cjFyjdbT --output eva_clip_model.pth
fi

read -p "VideoMAEv2 (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading VideoMAEv2"
    gdown https://drive.google.com/uc?id=1ftR-tZgHq4aU6dLKyGh9XtAGpZXCtV_7 --output videoMAEv2_model.pth
fi

read -p "VideoCLIP (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading VideoCLIP"
##
fi

read -p "HMDB-51 dataset (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading HMDB-51 dataset"
    gdown https://drive.google.com/uc?id=1SeLNhVD92qqE0MaQAZOEIyF5uq3a2wKS --output hmdb51_mp4.zip
    unzip hmdb51_mp4.zip
    mv hmdb51_mp4/hmdb51_mp4 hmdb51_mp42
    rm -rf hmdb51_mp4
    mv hmdb51_mp42 hmdb51_mp4
    ./create_pretrain_csv.sh official_hmdb_splits1/train.csv
    rm hmdb51_mp4.zip
    mv hmdb51_mp4 hmdb51_mp42
    mkdir hmdb51_mp4
    mv hmdb51_mp42 hmdb51_mp4/hmdb51_mp4
fi

read -p "Kinetics-400 TINY dataset (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading Kinetics-400 TINY dataset"
    gdown https://drive.google.com/uc?id=1dn4EJnCxzDIbB51yeAREwmoEm9pIzAAF
    unzip tiny-Kinetics-400.zip
    ./create_pretrain_csv.sh official_tiny_splits1/train.csv
    rm tiny-Kinetics-400.zip
    mv tiny-Kinetics-400 tiny-Kinetics-4002
    mkdir tiny-Kinetics-400
    mv tiny-Kinetics-4002 tiny-Kinetics-400/tiny-Kinetics-400
fi

read -p "Kinetics-400 dataset (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading Kinetics-400 dataset"
    # Your specific action for "yes" on the fifth question goes here
fi

read -p "how_to_100M dataset (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "downloading how_to_100M dataset"
    # Your specific action for "yes" on the fifth question goes here
fi



