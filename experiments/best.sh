#!/usr/bin/env bash

dataset="${1:-twitter}"
model="${2:-dcnn}"
num_runs=1
optim=adagrad

# twitter uses defaults from paper and/or https://github.com/FredericGodin/DynamicCNN
[[ $dataset =~ twitter.* ]] && num_epochs=8 || num_epochs=15
# our results seem to indicate 32 is also good for twitter; paper uses 4
[[ $dataset =~ twitter.* ]] && batch_size=32 || batch_size=128
# our results seem to indicate .05 is also good for twitter; paper uses .1
[[ $dataset =~ twitter.* ]] && lr=.05 || lr=.04
# our results seem to indicate 80 is also good for twitter; paper uses 60
[[ $dataset =~ twitter.* ]] && embed_dim=80 || embed_dim=100
[[ $dataset =~ twitter.* ]] && eval_period=300 || eval_period=400

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
