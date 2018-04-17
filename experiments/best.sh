#!/usr/bin/env bash

dataset="${1:-twitter}"
model="${2:-dcnn}"
num_runs=1
optim=adagrad

# twitter uses defaults from paper and/or https://github.com/FredericGodin/DynamicCNN
[[ $dataset =~ twitter.* ]] && num_epochs=8 || num_epochs=10
# we use 16; paper uses 4
[[ $dataset =~ twitter.* ]] && batch_size=16 || batch_size=64
# we use .03; paper uses .1
[[ $dataset =~ twitter.* ]] && lr=.03 || lr=.02
[[ $dataset =~ twitter.* ]] && eval_period=200 || eval_period=400
# we use 80; paper uses 60; must be: k * 2**num_layers, k = an int multiple of 2
[[ $dataset =~ twitter.* ]] && embed_dim=80 || embed_dim=128
[[ $dataset =~ twitter.* ]] && kernel_sizes="7 5" || kernel_sizes="11 9 7 5 3"
[[ $dataset =~ twitter.* ]] && num_filters="6 14" || num_filters="10 14 18 22 26"

for i in $(seq 1 $num_runs)
do
	python3 dcnn.py \
		--current-run $i \
		--model $model \
		--num-epochs $num_epochs \
		--batch-size $batch_size \
		--lr $lr \
		--optim $optim \
		--dataset $dataset \
		--embed-dim $embed_dim \
		--eval-period $eval_period \
		--kernel-sizes $kernel_sizes \
		--num-filters $num_filters \
		--log "logs/$dataset/best_$model"
done
