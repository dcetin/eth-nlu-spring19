from data_loader import DataLoader
from experiment_runner import ExperimentRunner

# initialize random seed
import numpy as np
from tensorflow import set_random_seed
np.random.seed(13)
set_random_seed(13)

loader = DataLoader()
data_train = loader.get_data('sct_train.csv')
data_eval = loader.get_data('sct_val.csv')

runner = ExperimentRunner(output_dir='output', vocabulary=loader.get_tokenizer().word_index)
#runner.list_models() # lists all models
#runner.list_experiments() # lists all experiments

#model = runner.new_experiment('experiment-1', 'model-1', params={num_layers: 3})
#model = runner.get_experiment('experiment-1') # load
#model.train(data_train, data_eval, n_epochs=10) # train, save checkpoint (keep number of epochs) compute accuracy for story close task and for sentiment
#model.evaluate(data_test) # evaluation: perplexities sent1 ... sent4 sent_real sent_fake, sentiment sent1 ... sent5 sent_real sent_fake
#model.predict(data_test) # produce file with 1 or 2 per line for each story
