#!/bin/bash

dataset="${1:-twitter}"
sizes=(4 8 16 32 64 128 256)

for batch_size in "${sizes[@]}"
do
    python3 dcnn.py --batch-size $batch_size --eval-period $((800 / $batch_size)) --dataset $dataset --log "logs/$dataset/batch-size-$batch_size"
done
