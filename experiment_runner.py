from models import model_zoo
import os
from tensorflow import keras
from tensorflow.keras.utils import Progbar
from scipy.special import softmax
from tensorflow.keras.utils import to_categorical
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

def evaluate(model, stories, sentiments, batch_size=50, return_hidden=False):
    batches_stories, counts_stories = get_batches(stories, batch_size, return_remainder=True)
    batches_sentiments, counts_sentiments = get_batches(sentiments, batch_size, return_remainder=True)

    pb = Progbar(batches_stories.shape[0])
    pb.update(0)

    perplexities = np.zeros(batches_stories.shape[:-1])
    proba_es = np.zeros(batches_stories.shape[:-1])
    proba_e = np.zeros(batches_stories.shape[:-1])
    sentiments_correct = np.zeros(batches_stories.shape[:-1], dtype='int32')
    hidden_out = None # instantiate lazily

    for bn, (stories_batch, sentiments_batch) in enumerate(zip(batches_stories, batches_sentiments)):
        losses = list()

        # compute P(sentence_n|sentence_(n-1)...sentence_1)
        model.reset_states() # reset state for story
        for i in range(stories_batch.shape[1]):
            sequences = stories_batch[:, i]
            sentiments = sentiments_batch[:, i]
            X, Y = get_XY_pair(sequences)
            Y_pred_proba, sentiments_proba, hidden = model.predict_on_batch(X)
            perplexities[bn, :, i] = calculate_perplexities(Y, Y_pred_proba)
            proba_es[bn, :, i] = calculate_probabs(Y, Y_pred_proba)
            sentiments_pred = np.argmax(sentiments_proba, axis=1)
            sentiments_correct[bn, :, i] = (sentiments.squeeze() == sentiments_pred)
            if return_hidden:
                if hidden_out is None:
                    hidden_out_shape = tuple(batches_stories.shape[:-1]) + (hidden.shape[-1],)
                    hidden_out = np.zeros(hidden_out_shape)
                hidden_out[bn, :, i] = hidden

        # compute P(sentence_n)
        for i in range(stories_batch.shape[1]):
            model.reset_states() # reset state for each turn
            sequences = stories_batch[:, i]
            X, Y = get_XY_pair(sequences)
            Y_pred_proba, _, _ = model.predict_on_batch(X)
            #norm_probabs[bn, :, i] = np.exp(np.log(norm_probabs[bn, :, i]) - np.log(calculate_probabs(Y, Y_pred_proba)))
            proba_e[bn, :, i] = calculate_probabs(Y, Y_pred_proba)

        pb.update(bn+1)

    perplexities = perplexities.reshape(-1, stories.shape[1])
    sentiments_correct = sentiments_correct.reshape(-1, stories.shape[1])
    proba_e = proba_e.reshape(-1, stories.shape[1])
    proba_es = proba_es.reshape(-1, stories.shape[1])
    total_counts = np.sum(counts_stories)
    proba_ratio = np.exp(np.log(proba_es) - np.log(proba_e))
    if return_hidden:
        hidden_out = hidden_out.reshape(-1, stories.shape[1], hidden_out.shape[-1])
        return (perplexities[:total_counts],
            proba_ratio[:total_counts],
            sentiments_correct[:total_counts],
            proba_es[:total_counts],
            proba_e[:total_counts],
            hidden_out[:total_counts]
        )
    else:
        return (perplexities[:total_counts],
            proba_ratio[:total_counts],
            sentiments_correct[:total_counts],
            proba_es[:total_counts],
            proba_e[:total_counts]
        )

class ModelWrapper:
    def __init__(self, output_dir, global_step, model):
        self.model = model
        self.output_dir = output_dir
        self.global_step = global_step
        self.batch_size = model.input.shape[0]

    def evaluate(self, data_eval, output_prefix, limit=None):
        y = data_eval['stories_correct'][:limit]

        ppls_one, norm_probabs_one, corr_pols_one, _, _ = evaluate(self.model, data_eval['stories_one'][:limit], data_eval['sentiment_one'][:limit])
        ppls_two, norm_probabs_two, corr_pols_two, _, _ = evaluate(self.model, data_eval['stories_two'][:limit], data_eval['sentiment_two'][:limit])

        ppls_result = ppls_one[:, -1].ravel() < ppls_two[:, -1].ravel()
        probabs_result = norm_probabs_one[:, -1].ravel() > norm_probabs_two[:, -1].ravel()

        y_pred_ppls = (2-ppls_result.astype(int))
        y_pred_probabs = (2-probabs_result.astype(int))

        ppls_accuracy = np.count_nonzero(y==y_pred_ppls)/y.shape[0]
        probabs_accuracy = np.count_nonzero(y==y_pred_probabs)/y.shape[0]

        corr_pols = np.concatenate([corr_pols_one[:, :4], corr_pols_one[:, 4, None], corr_pols_two[:, 4, None]], axis=1).ravel()
        sentiment_accuracy = np.count_nonzero(corr_pols)/corr_pols.shape[0]

        with open(os.path.join(self.output_dir, '%s-evaluate-accuracy.tsv' % output_prefix), 'w') as f:
            f.write('type\tscore\n')
            f.write('%s\t%f\n' % ('accuracy-on-pplty', ppls_accuracy))
            f.write('%s\t%f\n' % ('accuracy-on-proba_ratio', probabs_accuracy))
            f.write('%s\t%f\n' % ('sentiment_accuracy', sentiment_accuracy))

        output = np.concatenate([
            ppls_one[:, :4], ppls_one[:, 4, None], ppls_two[:, 4, None],
        ], axis=1)
        header = '\t'.join(['ppl1', 'ppl2', 'ppl3', 'ppl4', 'ppl5a', 'ppl5b'])
        np.savetxt(os.path.join(self.output_dir, '%s-evaluate-pplty.tsv' % output_prefix),
            output, header=header, delimiter='\t')

        output = np.concatenate([
            norm_probabs_one[:, :4], norm_probabs_one[:, 4, None], norm_probabs_two[:, 4, None],
        ], axis=1)
        header = '\t'.join(['proba_ratio1', 'proba_ratio2', 'proba_ratio3', 'proba_ratio4', 'proba_ratio5a', 'proba_ratio5b'])
        np.savetxt(os.path.join(self.output_dir, '%s-evaluate-proba_ratio.tsv' % output_prefix),
            output, header=header, delimiter='\t')

        output = np.concatenate([
            corr_pols_one[:, :4], corr_pols_one[:, 4, None], corr_pols_two[:, 4, None]
        ], axis=1)
        header = '\t'.join(['corr_pol1', 'corr_pol2', 'corr_pol3', 'corr_pol4', 'corr_pol5a', 'corr_pol5b'])
        np.savetxt(os.path.join(self.output_dir, '%s-evaluate-sentiment.tsv' % output_prefix),
            output, header=header, delimiter='\t', fmt='%d')

    def predict(self, data_eval, output_prefix, limit=None):
        ppls_one, norm_probabs_one, _, _, _ = evaluate(self.model, data_eval['stories_one'][:limit], data_eval['sentiment_one'][:limit])
        ppls_two, norm_probabs_two, _, _, _ = evaluate(self.model, data_eval['stories_two'][:limit], data_eval['sentiment_two'][:limit])

        ppls_result = ppls_one[:, -1].ravel() < ppls_two[:, -1].ravel()
        probabs_result = norm_probabs_one[:, -1].ravel() > norm_probabs_two[:, -1].ravel()

        predictions_ppls = (2-ppls_result.astype(int))
        predictions_probabs = (2-probabs_result.astype(int))

        np.savetxt(os.path.join(self.output_dir, '%s-predictions-on-pplty.tsv' % output_prefix),
            predictions_ppls, delimiter='\t', fmt='%d')

        np.savetxt(os.path.join(self.output_dir, '%s-predictions-on-proba_ratio.tsv' % output_prefix),
            predictions_probabs, delimiter='\t', fmt='%d')

    def transform(self, data_eval, output_prefix, limit=None):
        _, proba_ratio_one, _, proba_es_one, proba_e_one, hidden_one = evaluate(self.model, data_eval['stories_one'][:limit], data_eval['sentiment_one'][:limit], return_hidden=True)
        _, proba_ratio_two, _, proba_es_two, proba_e_two, hidden_two = evaluate(self.model, data_eval['stories_two'][:limit], data_eval['sentiment_two'][:limit], return_hidden=True)

        all_features = np.concatenate([
            np.log(proba_ratio_one[:, -1, None]),
            np.log(proba_es_one[:, -1, None]),
            np.log(proba_e_one[:, -1, None]),
            np.log(proba_ratio_two[:, -1, None]),
            np.log(proba_es_two[:, -1, None]),
            np.log(proba_e_two[:, -1, None]),
            to_categorical(data_eval['sentiment_one'][:limit, -1], 3),
            to_categorical(data_eval['sentiment_two'][:limit, -1], 3)
        ], axis=1)

        header = 'one_proba_ratio\tone_proba_es\tone_proba_e\ttwo_proba_ratio\ttwo_proba_es\ttwo_proba_e\tone_sent_neg\tone_sent_neu\tone_sent_pos\ttwo_sent_neg\ttwo_sent_neu\ttwo_sent_pos'
        np.savetxt(os.path.join(self.output_dir, '%s-transform-features.tsv' % output_prefix),
            all_features, delimiter='\t', header=header)

        all_hidden = np.concatenate([hidden_one[:, -1], hidden_two[:, -1]], axis=1)
        np.save(os.path.join(self.output_dir, '%s-transform-hidden.npy' % output_prefix), all_hidden)

        if 'stories_correct' in data_eval:
            all_labels = data_eval['stories_correct'][:limit]
            header = 'correct_ending'
            np.savetxt(os.path.join(self.output_dir, '%s-transform-labels.tsv' % output_prefix),
                all_labels, delimiter='\t', fmt='%d', header=header)

    def train(self, data_train, data_eval, max_epochs=10, limit=None, eval_each_epoch=1, checkpoint_each_epoch=1):
        n_epochs = max_epochs - self.global_step
        if n_epochs <= 0:
            print('*** already trained for %d epochs - skipping training ***' % self.global_step)
            return

        y = data_eval['stories_correct'][:limit]
        for ep in range(self.global_step, max_epochs):
            print('Epoch %d/%d' % (ep+1, max_epochs))

            loss = train_one_epoch(self.model, data_train['stories_one'][:limit], data_train['sentiment_one'][:limit])

            if ep % checkpoint_each_epoch == 0:
                ckpt_path = os.path.join(self.output_dir, 'model_weights-%d.h5' % (ep+1))
                self.model.save_weights(ckpt_path)
                print('*** saved model weights for epoch %d ***' % (ep+1))

            if ep % eval_each_epoch == 0:
                ppls_one, norm_probabs_one, corr_pols_one, _, _ = evaluate(self.model, data_eval['stories_one'][:limit], data_eval['sentiment_one'][:limit])
                ppls_two, norm_probabs_two, corr_pols_two, _, _ = evaluate(self.model, data_eval['stories_two'][:limit], data_eval['sentiment_two'][:limit])

                corr_pols = np.concatenate([corr_pols_one[:, :4], corr_pols_one[:, 4, None], corr_pols_two[:, 4, None]], axis=1).ravel()
                sentiment_accuracy = np.count_nonzero(corr_pols)/corr_pols.shape[0]

                ppls_result = ppls_one[:, -1].ravel() < ppls_two[:, -1].ravel()
                probabs_result = norm_probabs_one[:, -1].ravel() > norm_probabs_two[:, -1].ravel()

                y_pred_ppls = (2-ppls_result.astype(int))
                y_pred_probabs = (2-probabs_result.astype(int))

                ppls_accuracy = np.count_nonzero(y==y_pred_ppls)/y.shape[0]
                probabs_accuracy = np.count_nonzero(y==y_pred_probabs)/y.shape[0]

            mode = 'w' if ep == 0 else 'a'
            with open(os.path.join(self.output_dir, 'training-report.tsv'), mode) as fout:
                if ep == 0:
                    fout.write('# epoch\taccuracy-on-pplty\taccuracy-on-proba_ratio\tsentiment_accuracy\tloss\n')
                if ep % eval_each_epoch == 0:
                    fout.write('%d\t%f\t%f\t%f\t%f\n' % (ep+1, ppls_accuracy, probabs_accuracy, sentiment_accuracy, loss))
                else:
                    fout.write('%d\tn/a\tn/a\tn/a\t%f\n' % (ep+1, loss))

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

    def list_models(self):
        return list(model_zoo.keys())

    def get_experiment(self, experiment_name, epoch=None):
        output_dir = os.path.join(self.output_dir, '%s' % experiment_name)
        with open(os.path.join(output_dir, 'settings.json'), 'r') as f:
            settings = json.load(f)
        if epoch is None:
            with open(os.path.join(output_dir, 'global_step'), 'r') as f:
                global_step = int(f.read())
        else:
            global_step = epoch
        model = model_zoo[settings['model_name']](self.vocabulary, **settings['params'])
        try:
            if epoch:
                model.load_weights(os.path.join(output_dir, 'model_weights-%d.h5' % epoch))
                print('*** loaded model weights for epoch %d ***' % epoch)
            else:
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
