requirements:
	python -m pip install -r requirements.txt --user
	./download-data.sh

train:
	bsub -W 04:00 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" python run_experiments.py \
	--new "sample-experiment" --model "default_model"  --train-for 5 \
	--params "{\"num_layers\": 1, \"hidden_size\": 512, \"loss_weights\": [0.7, 0.3, 0]}"

eval:
	bsub -W 04:00 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" python run_experiments.py \
	--load "sample-experiment" --checkpoint 2 --evaluate-all --predict-all --transform-all

classifiers:
	bsub -W 04:00 -n 4 -R "rusage[mem=2048,ngpus_excl_p=1]" python run_experiments.py \
	--load "sample-experiment" --classifiers

.PHONY: requirements train eval classifiers