#!/usr/bin/env bash

dataset="${1:-twitter}"
models=(dcnn dcnn-custom mlp)

for model in "${models[@]}"
do
    [[ $model = "dcnn-custom" ]] && export CUDA_VISIBLE_DEVICES=-1 || export CUDA_VISIBLE_DEVICES=0
	python3 dcnn.py --model $model --num-epochs 20 --batch-size 256 --dataset $dataset --log "logs/$dataset/model-$model"
done
