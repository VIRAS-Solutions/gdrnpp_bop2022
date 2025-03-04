#!/bin/bash

xhost +

# prepare /datasets, /pretrained_models and /output folders as explained in the main README.md
gdrnpp_dir=${PWD%/*}

docker run \
--gpus all \
--net=host \
-it --rm \
--shm-size=32gb --env="DISPLAY" \
--volume="${gdrnpp_dir}:/gdrnpp_bop2022" \
--volume="/home/dftlabor/Schreibtisch/Leon/datasets:/gdrnpp_bop2022/datasets" \
--volume="/home/dftlabor/Schreibtisch/Leon/pretrained_models:/gdrnpp_bop2022/pretrained_models" \
--volume="/home/dftlabor/Schreibtisch/Leon/output:/gdrnpp_bop2022/output" \
--name=gdrnppv0 gdrnpp


xhost -
