from models import model_zoo
import os
from tensorflow import keras
from tensorflow.keras.utils import Progbar
from scipy.special import softmax
import tensorflow as tf
import numpy as np
import pickle
import json

def get_XY_pair(data):
    X = data[:, :-1]
    mask = X.copy()
    mask[np.nonzero(mask)] = 1
    Y = mask * data[:, 1:]
    return X, Y

def calculate_perplexities(Y, Y_pred):
    perplexities = []
    for j in range(Y.shape[0]):
        N = np.count_nonzero(Y[j])
        if N > 0:
            s = 0
            for pos in range(Y.shape[1]):
                if Y[j][pos] > 0:
                    s += np.log(Y_pred[j][pos][Y[j][pos]])
            perplexities.append(np.exp(-1/N * s))
        else:
            perplexities.append(np.nan)
    return np.array(perplexities)

def calculate_probabs(Y, Y_pred):
    probabs = []
    for j in range(Y.shape[0]):
        N = np.count_nonzero(Y[j])
        if N > 0:
            s = 0
            for pos in range(Y.shape[1]):
                if Y[j][pos] > 0:
                    s += np.log(Y_pred[j][pos][Y[j][pos]])
            probabs.append(np.exp(s))
        else:
            probabs.append(np.nan)
    return np.array(probabs)

def get_batches_generator(stories, batch_size, return_remainder):
    """ Returns batches of the form (BATCH_SIZE, NUM_SENTS, NUM_WORDS),
        where
            NUM_SENTS = sents_per_story
            BATCH_SIZE: number of sentences per batch (multiple of NUM_SENTS)
            NUM_WORDS: number of words in sentence (MAX_TIME_STEPS+1)
        for training, the state of the stateful LSTM can then be reset after
        each group
    """

    n_stories = stories.shape[0]
    sents_per_story = stories.shape[1]
    try:
        n_words = stories.shape[2]
    except:
        n_words = 1
    n_samples = n_stories*sents_per_story

    sequences = stories.reshape(-1, n_words)

    multiplier, remainder = np.divmod(batch_size, sents_per_story)
    if remainder != 0:
        raise ValueError('batch_size must be multiple of sentences_per_story')

    bs = sents_per_story * multiplier
    for i in range(0, n_samples, sents_per_story*bs):
        X = sequences[i:i + sents_per_story*bs]
        if X.shape[0] == sents_per_story*bs:
            yield X.reshape(bs, sents_per_story, n_words), X.shape[0] / sents_per_story
        else:
            # handle non-batchsize-multiple
            if return_remainder:
                padded = np.zeros(shape=(sents_per_story*bs, n_words))
                padded[:X.shape[0]] = X
                yield padded.reshape(bs, sents_per_story, n_words), X.shape[0] / sents_per_story
            else:
                pass

def get_batches(stories, batch_size, return_remainder):
    list_data = []
    list_count = []
    for data, count in get_batches_generator(stories, batch_size, return_remainder):
        list_data.append(data)
        list_count.append(count)
    return np.array(list_data, dtype='int32'), np.array(list_count, dtype='int32')

def train_one_epoch(model, stories, sentiments, batch_size=50, shuffle=True):

    # get an index to shuffle the data
    if shuffle:
        index = np.arange(stories.shape[0])
        np.random.shuffle(index)
        stories = stories[index]
        sentiments = sentiments[index]

    batches_stories, counts_stories = get_batches(stories, batch_size, return_remainder=False)
    batches_sentiments, counts_sentiments = get_batches(sentiments, batch_size, return_remainder=False)

    pb = Progbar(batches_stories.shape[0])
    pb.update(0)

    losses_per_batch = list()

    for bn, (stories_batch, sentiments_batch) in enumerate(zip(batches_stories, batches_sentiments)):
        losses = list()
        for i in range(stories_batch.shape[1]):
            sequences = stories_batch[:, i]
            sentiments = sentiments_batch[:, i]
            X, Y = get_XY_pair(sequences)

            loss = model.train_on_batch(X, [Y, sentiments])
            losses.append(loss)

        pb.update(bn+1, [('loss', np.mean(losses))])
        losses_per_batch.append(np.mean(losses))

        # reset state
        model.reset_states()
    return np.mean(losses_per_batch)

def evaluate(model, stories, sentiments, batch_size=50):
    batches_stories, counts_stories = get_batches(stories, batch_size, return_remainder=True)
    batches_sentiments, counts_sentiments = get_batches(sentiments, batch_size, return_remainder=True)

    pb = Progbar(batches_stories.shape[0])
    pb.update(0)

    perplexities = np.zeros(batches_stories.shape[:-1])
    norm_probabs = np.zeros(batches_stories.shape[:-1])
    sentiments_correct = np.zeros(batches_stories.shape[:-1], dtype='int32')

    for bn, (stories_batch, sentiments_batch) in enumerate(zip(batches_stories, batches_sentiments)):
        losses = list()

        # compute P(sentence_n|sentence_(n-1)...sentence_1)
        model.reset_states() # reset state for story
        for i in range(stories_batch.shape[1]):
            sequences = stories_batch[:, i]
            sentiments = sentiments_batch[:, i]
            X, Y = get_XY_pair(sequences)
            Y_pred_proba, sentiments_proba, _ = model.predict_on_batch(X)
            perplexities[bn, :, i] = calculate_perplexities(Y, Y_pred_proba)
            norm_probabs[bn, :, i] = calculate_probabs(Y, Y_pred_proba)
            sentiments_pred = np.argmax(sentiments_proba, axis=1)
            sentiments_correct[bn, :, i] = (sentiments.squeeze() == sentiments_pred)

        # compute P(sentence_n) and normalize results
        for i in range(stories_batch.shape[1]):
            model.reset_states() # reset state for each turn
            sequences = stories_batch[:, i]
            X, Y = get_XY_pair(sequences)
            Y_pred_proba, _, _ = model.predict_on_batch(X)
            norm_probabs[bn, :, i] = np.exp(np.log(norm_probabs[bn, :, i]) - np.log(calculate_probabs(Y, Y_pred_proba)))

        pb.update(bn+1)

    perplexities = perplexities.reshape(-1, stories.shape[1])
    sentiments_correct = sentiments_correct.reshape(-1, stories.shape[1])
    norm_probabs = norm_probabs.reshape(-1, stories.shape[1])
    total_counts = np.sum(counts_stories)

    return perplexities[:total_counts], norm_probabs[:total_counts], sentiments_correct[:total_counts]

class ModelWrapper:
    def __init__(self, output_dir, global_step, model):
        self.model = model
        self.output_dir = output_dir
        self.global_step = global_step
        self.batch_size = model.input.shape[0]

    def evaluate(self, data_eval, output_suffix, limit=None):
        ppls_real, norm_probabs_real, corr_pols_real = evaluate(self.model, data_eval['stories_real'][:limit], data_eval['sentiment_real'][:limit])
        ppls_fake, norm_probabs_fake, corr_pols_fake = evaluate(self.model, data_eval['stories_fake'][:limit], data_eval['sentiment_fake'][:limit])

        ppls_result = ppls_real[:, -1].ravel() < ppls_fake[:, -1].ravel()
        ppls_accuracy = np.count_nonzero(ppls_result)/ppls_result.shape[0]
        probabs_result = norm_probabs_real[:, -1].ravel() > norm_probabs_fake[:, -1].ravel()
        probabs_accuracy = np.count_nonzero(probabs_result)/probabs_result.shape[0]
        with open(os.path.join(self.output_dir, 'evaluate-accuracy-%s.tsv' % output_suffix), 'w') as f:
            f.write('type\tscore\n')
            f.write('%s\t%f\n' % ('accuracy-on-pplty', ppls_accuracy))
            f.write('%s\t%f\n' % ('accuracy-on-probab_ratio', probabs_accuracy))

        output = np.concatenate([
            ppls_real[:, :4], ppls_real[:, 4, None], ppls_fake[:, 4, None],
        ], axis=1)
        header = '\t'.join(['ppl1', 'ppl2', 'ppl3', 'ppl4', 'ppl5a', 'ppl5b'])
        np.savetxt(os.path.join(self.output_dir, 'evaluate-pplty-%s.tsv' % output_suffix),
            output, header=header, delimiter='\t')

        output = np.concatenate([
            norm_probabs_real[:, :4], norm_probabs_real[:, 4, None], norm_probabs_fake[:, 4, None],
        ], axis=1)
        header = '\t'.join(['probab_ratio1', 'probab_ratio2', 'probab_ratio3', 'probab_ratio4', 'probab_ratio5a', 'probab_ratio5b'])
        np.savetxt(os.path.join(self.output_dir, 'evaluate-probab_ratio-%s.tsv' % output_suffix),
            output, header=header, delimiter='\t')

        output = np.concatenate([
            corr_pols_real[:, :4], corr_pols_real[:, 4, None], corr_pols_fake[:, 4, None]
        ], axis=1)
        header = '\t'.join(['corr_pol1', 'corr_pol2', 'corr_pol3', 'corr_pol4', 'corr_pol5a', 'corr_pol5b'])
        np.savetxt(os.path.join(self.output_dir, 'evaluate-sentiment-%s.tsv' % output_suffix),
            output, header=header, delimiter='\t', fmt='%d')

    def predict(self, data_eval, output_suffix, limit=None):
        ppls_real, norm_probabs_real, _ = evaluate(self.model, data_eval['stories_real'][:limit], data_eval['sentiment_real'][:limit])
        ppls_fake, norm_probabs_fake, _ = evaluate(self.model, data_eval['stories_fake'][:limit], data_eval['sentiment_fake'][:limit])

        ppls_result = ppls_real[:, -1].ravel() < ppls_fake[:, -1].ravel()
        probabs_result = norm_probabs_real[:, -1].ravel() > norm_probabs_fake[:, -1].ravel()

        predictions_ppls = (2-ppls_result.astype(int))
        predictions_probabs = (2-probabs_result.astype(int))

        np.savetxt(os.path.join(self.output_dir, 'predictions-on-pplty-%s.tsv' % output_suffix),
            predictions_ppls, delimiter='\t', fmt='%d')

        np.savetxt(os.path.join(self.output_dir, 'predictions-on-probab_ratio-%s.tsv' % output_suffix),
            predictions_probabs, delimiter='\t', fmt='%d')

    def train(self, data_train, data_eval, max_epochs=10, limit=None, eval_each_epoch=1):
        n_epochs = max_epochs - self.global_step
        if n_epochs <= 0:
            print('*** skipping training ***')
            return

        for ep in range(self.global_step, max_epochs):
            print('Epoch %d/%d' % (ep+1, max_epochs))

            loss = train_one_epoch(self.model, data_train['stories_real'][:limit], data_train['sentiment_real'][:limit])

            if ep % eval_each_epoch == 0:
                ppls_real, norm_probabs_real, _ = evaluate(self.model, data_eval['stories_real'][:limit], data_eval['sentiment_real'][:limit])
                ppls_fake, norm_probabs_fake, _ = evaluate(self.model, data_eval['stories_fake'][:limit], data_eval['sentiment_fake'][:limit])

                ppls_result = ppls_real[:, -1].ravel() < ppls_fake[:, -1].ravel()
                ppls_accuracy = np.count_nonzero(ppls_result)/ppls_result.shape[0]

                probabs_result = norm_probabs_real[:, -1].ravel() > norm_probabs_fake[:, -1].ravel()
                probabs_accuracy = np.count_nonzero(probabs_result)/probabs_result.shape[0]

            mode = 'w' if ep == 0 else 'a'
            with open(os.path.join(self.output_dir, 'training-report.tsv'), mode) as fout:
                if ep == 0:
                    fout.write('# epoch\taccuracy-on-pplty\taccuracy-on-probab_ratio\tloss\n')
                if ep % eval_each_epoch == 0:
                    fout.write('%d\t%f\t%f\t%f\n' % (ep+1, ppls_accuracy, probabs_accuracy, loss))
                else:
                    fout.write('%d\tn/a\tn/a\t%f\n' % (ep+1, loss))

        model_path = os.path.join(self.output_dir, 'model_weights.h5')
        self.model.save_weights(model_path)
        print('*** saved model weights ***')

        self.global_step += n_epochs
        with open(os.path.join(self.output_dir, 'global_step'), 'w') as f:
            f.write('%d' % self.global_step)


class ExperimentRunner:
    def __init__(self, output_dir, vocabulary):
        self.output_dir = output_dir
        self.vocabulary = vocabulary

    def list_experiments(self):
        return os.listdir(self.output_dir)

    def get_experiment(self, experiment_name):
        output_dir = os.path.join(self.output_dir, '%s' % experiment_name)
        #model = keras.models.load_model(model_path)
        with open(os.path.join(output_dir, 'settings.json'), 'r') as f:
            settings = json.load(f)
        with open(os.path.join(output_dir, 'global_step'), 'r') as f:
            global_step = int(f.read())
        model = model_zoo[settings['model_name']](self.vocabulary, **settings['params'])
        try:
            model.load_weights(os.path.join(output_dir, 'model_weights.h5'))
            print('*** loaded model weights ***')
        except:
            pass
        return ModelWrapper(output_dir, global_step, model)

    def new_experiment(self, experiment_name, model_name, **params):
        output_dir = os.path.join(self.output_dir, '%s' % experiment_name)
        model_path = os.path.join(output_dir, 'model.h5')
        os.makedirs(output_dir)
        model = model_zoo[model_name](self.vocabulary, **params)
        #model.save(model_path)
        with open(os.path.join(output_dir, 'settings.json'), 'w') as f:
            json.dump(dict(model_name=model_name,
                params=params), f, indent=4)
        with open(os.path.join(output_dir, 'global_step'), 'w') as f:
            f.write('0')
        return ModelWrapper(output_dir, 0, model)
