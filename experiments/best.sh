#!/usr/bin/env bash

dataset="${1:-twitter}"
model=dcnn
num_runs=3
optim=adagrad

# twitter uses defaults from paper and/or https://github.com/FredericGodin/DynamicCNN
[[ $dataset = "twitter" ]] && num_epochs=5 || num_epochs=10
# our results seem to indicate 32 is also good for twitter
[[ $dataset = "twitter" ]] && batch_size=4 || batch_size=128
# our results seem to indicate .05 is also good for twitter
[[ $dataset = "twitter" ]] && lr=.1 || lr=.05
# our results seem to indicate 80 is also good for twitter
[[ $dataset = "twitter" ]] && embed_dim=48 || embed_dim=160
[[ $dataset = "twitter" ]] && eval_period=100 || eval_period=200

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
		--log "logs/$dataset/best_$model"
done