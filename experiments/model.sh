#!/usr/bin/env bash

dataset="${1:-twitter}"
models=(dcnn mlp)

for model in "${models[@]}"
do
  python3 dcnn.py --model $model --dataset $dataset --log "logs/$dataset/model-$model"
done
