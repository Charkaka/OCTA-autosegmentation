#!/bin/bash

docker run \
    --shm-size=8g \
    --gpus all \
    --mount type=bind,source=$(pwd)/datasets/,target=/var/dataset/ \
    --mount type=bind,source=$(pwd)/OCTA500/,target=/var/OCTA500/ \
    --mount type=bind,source=$(pwd)/results/,target=/var/segmented/ \
    --mount type=bind,source=$(pwd)/generation/,target=/var/generation/ \
    --mount type=bind,source=$(pwd)/,target=/home/OCTA-seg/ \
    --mount type=bind,source=$(pwd)/docker/trained_models/new_GAN/,target=/var/docker/trained_models/new_GAN/ \
    octa-seg train