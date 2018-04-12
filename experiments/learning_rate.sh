#!/bin/bash

dataset="${1:-twitter}"
rates=(0.5 0.1 0.05 0.01 0.005 0.001 0.0005)

for lr in "${rates[@]}"
do
    python3 dcnn.py --num-epochs 25 --batch-size 32 --lr $lr --dataset $dataset --log "logs/$dataset/lr-$lr"
done
