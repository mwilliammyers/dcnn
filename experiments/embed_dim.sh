#!/bin/bash

dims=(20 40 60 80 100)
dataset="${1:-twitter}"

for d in "${dims[@]}"
do
    python3 dcnn.py --num-epochs 20 --batch-size 32 --lr 0.05 --embed-dim $d --dataset $dataset --log "logs/$dataset/embed-dim-$d"
done
