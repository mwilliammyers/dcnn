#!/usr/bin/env bash

dataset="${1:-twitter}"
models=(dcnn dcnn-relu dcnn-leakyrelu mlp)

for model in "${models[@]}"
do
	python3 dcnn.py --model $model --num-epochs 25 --batch-size 32 --dataset $dataset --log "logs/$dataset/model-$model" &
done

wait