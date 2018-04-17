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
[[ $dataset =~ twitter.* ]] && embed_dim=80 || embed_dim=180
[[ $dataset =~ twitter.* ]] && eval_period=200 || eval_period=400

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
