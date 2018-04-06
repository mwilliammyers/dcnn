#!/bin/bash

dims=(20 40 60 80 100)

for d in "${dims[@]}"
do
    python dcnn.py --num-epochs 20 --batch-size 32 --lr 0.05 --embed-dim $d --log "logs/embed-dim-$d"
done
