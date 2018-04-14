#!/usr/bin/env bash

dataset="${1:-twitter}"
models=(dcnn mlp)

for model in "${models[@]}"
do
	python3 dcnn.py --model $model --num-epochs 20 --batch-size 256 --dataset $dataset --log "logs/$dataset/model-$model"
done