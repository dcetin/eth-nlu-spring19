from data_loader import DataLoader
from experiment_runner import ExperimentRunner
import argparse
from sys import exit
import nltk

VOCABULARY_SIZE = 20000
MAX_SEQ_LEN = 91

# initialize random seed
import numpy as np
from tensorflow import set_random_seed
np.random.seed(13)
set_random_seed(13)

nltk.download('punkt')

argparser = argparse.ArgumentParser()
argparser.add_argument('-d', '--data-dir', action='store', dest='data_dir', help='here are all the data files stored', default='data')
argparser.add_argument('-w', '--working-dir', action='store', dest='working_dir', help='used to store temporary files', default='cache')
argparser.add_argument('-o', '--results-dir', action='store', dest='results_dir', help='here all the result files are created', default='output')
argparser.add_argument('-l', '--list-experiments', action='store_true', dest='list_experiments', help='list all available experiments')
argparser.add_argument('--load', action='store', dest='load_experiment', help='loads an existing experiment')
argparser.add_argument('--new', action='store', dest='new_experiment', help='creates a new experiment')
argparser.add_argument('--epochs', action='store', dest='epochs', help='number of epochs', default=10, type=int)
argparser.add_argument('--limit', action='store', dest='limit', help='limit processing to n samples', default=None, type=int)
argparser.add_argument('--model', action='store', dest='model_name', help='specifies the model')
argparser.add_argument('--params', action='store', dest='model_params', help='specifies parameters as json')
argparser.add_argument('--train-for', action='store', dest='train_for', help='train for n epochs', type=int)
argparser.add_argument('--evaluate-all', action='store_true', dest='evaluate_all', help='evaluate_all')
argparser.add_argument('--predict-all', action='store_true', dest='predict_all', help='predict_all')

args = argparser.parse_args()

loader = DataLoader(
    data_dir=args.data_dir,
    cache_dir=args.working_dir,
    max_vocabulary_size=VOCABULARY_SIZE,
    max_seq_len=MAX_SEQ_LEN)

data_train = loader.get_data('sct_train.csv')
data_eval = loader.get_data('sct_val.csv')
data_test_wlabel = loader.get_data('test_for_report-stories_labels.csv')
data_test = loader.get_data('test-stories.csv')

runner = ExperimentRunner(
    output_dir=args.results_dir,
    vocabulary=loader.get_tokenizer().word_index)

if args.list_experiments:
    print('\n'.join(runner.list_experiments()))
    exit(0)
elif args.load_experiment:
    model = runner.get_experiment(args.load_experiment)
elif args.new_experiment:
    model = runner.new_experiment(args.new_experiment, args.model_name, args.model_params)

if args.train_for:
     model.train(data_train, data_eval, max_epochs=args.train_for, limit=args.limit)

if args.evaluate_all:
    model.evaluate(data_eval, 'eval', limit=args.limit)
    model.evaluate(data_test_wlabel, 'test_wlabel', limit=args.limit)

if args.predict_all:
    model.predict(data_eval, 'eval', limit=args.limit)
    model.predict(data_test_wlabel, 'test_wlabel', limit=args.limit)
    model.predict(data_test, 'test_real', limit=args.limit)
