#!/bin/bash

dir=
sizes=(4 8 16 32 64 128 256)

for batch_size in "${sizes[@]}"
do
    python3 dcnn.py --num-epochs 10 --batch-size $batch_size --eval-period $((800 / $batch_size)) --log "logs/batch-size-$batch_size"
done
