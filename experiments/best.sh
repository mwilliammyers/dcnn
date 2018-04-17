#!/usr/bin/env bash

dataset="${1:-twitter}"
model="${2:-dcnn}"
num_runs=1
optim=adagrad

# twitter uses defaults from paper and/or https://github.com/FredericGodin/DynamicCNN
[[ $dataset =~ twitter.* ]] && num_epochs=8 || num_epochs=10
# our results seem to indicate 16 is also good for twitter; paper uses 4
[[ $dataset =~ twitter.* ]] && batch_size=16 || batch_size=64
# our results seem to indicate .03 is also good for twitter; paper uses .1
[[ $dataset =~ twitter.* ]] && lr=.03 || lr=.02
# our results seem to indicate 80 is also good for twitter; paper uses 60
[[ $dataset =~ twitter.* ]] && embed_dim=80 || embed_dim=256
[[ $dataset =~ twitter.* ]] && eval_period=200 || eval_period=200
[[ $dataset =~ twitter.* ]] && kernel_sizes="7 5" || kernel_sizes="7 7 5 5 3 3"
[[ $dataset =~ twitter.* ]] && num_filters="6 14" || num_filters="10 14 18 22 26 30"

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
