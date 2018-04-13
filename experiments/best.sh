#!/usr/bin/env bash

dataset="${1:-twitter}"
model=dcnn
[[ $dataset = "twitter" ]] && num_epochs=25 || num_epochs=20
[[ $dataset = "twitter" ]] && batch_size=32 || batch_size=128

python3 dcnn.py \
--model $model \
--num-epochs $num_epochs \
--batch-size $batch_size \
--dataset $dataset \
--log "logs/$dataset/best-$model"
