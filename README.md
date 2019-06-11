# Story Cloze Task

## Requirements

Environment can be set up by running the **requirements** recipe of the Makefile. Alternatively, Leonhard only requires installing **vaderSentiment** package on top if its **python_gpu/3.7.1** module.

```
nltk
tensorflow>=1.13.1
numpy
vaderSentiment
```

## Usage

Following examples can be simply run on Leonhard via the **train**, **eval** and **classifiers** recipes of the Makefile.

- **Running a new experiment:**
  This command creates the default model (Lambda rather than attention) with one GRU layer of 512 units with loss weighting of 7/3 and trains it for 3 epochs.
  
  ```bash
  python run_experiments.py --new "sample-experiment" --model "default_model" --params "{\"num_layers\": 1, \"hidden_size\": 512, \"loss_weights\": [0.7, 0.3, 0]}" --train-for 3
  ```
- **Continuing a previous experiment:**
  This command continues the previous experiment for two more epochs. If a checkpoint is not specified it will continue from the last epoch as default.
  ```bash
  python run_experiments.py --load "sample-experiment" --checkpoint 3 --train-for 5
  ```
- **Evaluating a finished experiment:**
  This command evaluates the finished experiment for the checkpoint with best validation result. Additional flags at the end respectively creates evaluation files that summarizes the results, produces predictions and transforms the validation dataset by extracting learned features. Instead of using `bestval` flag, another epoch can be provided via the `checkpoint` flag.
  ```bash
  python run_experiments.py --load "sample-experiment" --bestval --evaluate-all --predict-all --transform-all
  ```
- **Running classifiers on top of extracted features:**
  This command runs a set of classifiers on top of the features extracted from the validation dataset.
  
  ```bash
  python run_experiments.py --load "sample-experiment" --classifiers
  ```

## Structure

- `cache/` stores intermediary preprocessed data.
- `data/` stores the downloaded data and embedding files.
- `notebooks/` contains experimental notebooks.
- `output/` stores one directory per outputs of each experiment.
- `report/` contains the source LaTeX and pdf of the report.
- `classifiers.py` is the script that runs classifiers on extracted features.
- `data_loader.py` either loads cached files or downloads the required data.
- `download-data.sh` downloads the data in data directory.
- `experiment_runner.py` implements the experiment running pipeline.
- `models.py` implements the neural network models.
- `run_experiments.py` is the main script for running and evaluating experiments.
- `sct_dataset.py` implements the dataset reader class.
- `sentiment.py` has sentiment related functions.
- `tokenizer.py` implements the Tokenizer class.
- `util.py` has auxiliary helper functions.
- `README.md` is this manual.
- `requirements.txt` lists required packages.

## Outputs
- training-report.tsv: reports the progress of training (accuracies and loss)
- {dataset}-predictions-on-proba_ratio.tsv: prediction based on proba_ratio of last sentence (as in UW paper)
- {dataset}-predictions-on-ppty.tsv: prediction based on perplexity of last sentence
- {dataset}-evaluate-sentiment.tsv: logs sentiment prediction correct yes/no for each story and sentence
- {dataset}-evaluate-proba_ratio.tsv: reports proba_ratio for each story and sentence
- {dataset}-evaluate-ppty.tsv: reports perplexity for each story and sentence
- {dataset}-evaluate-accuracy.tsv: reports accuracies based on proba_ratio and ppty

{dataset} is a placeholder for "dev", "test", "report"; where "dev" is the validation set,
"test" is the test set WITH labels and "report" is the dataset we need to base our final predictions on.

## Contact

- **Thomas Diggelmann**, <thomasdi@student.ethz.ch>
- Doruk Cetin, <dcetin@student.ethz.ch>
- George Mtui, <gmtui@student.ethz.ch>