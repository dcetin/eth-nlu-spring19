import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

VADER_POLARITY_NEG = 0
VADER_POLARITY_POS = 2
VADER_POLARITY_NEU = 1
VADER_POLARITY_LABELS = ['neg', 'neu', 'pos']

def vader_polarity_scores(sentences):
    analyzer = SentimentIntensityAnalyzer()
    scores = np.zeros(shape=(len(sentences), 4), dtype='float32')
    for index, sentence in enumerate(sentences):
        vs = analyzer.polarity_scores(sentence)
        scores[index, 0] = vs['neg']
        scores[index, 1] = vs['neu']
        scores[index, 2] = vs['pos']
        scores[index, 3] = vs['compound']
    return scores

def vader_labelize_polarity(polarities_quantized):
    flat = polarities_quantized.ravel()
    return np.array([POLARITY_LABELS[p] for p in flat]).reshape(polarities_quantized.shape)

def vader_quantize_polarities(polarities, mu=0.05):
    pol = polarities[:, :, -1]
    output = np.empty_like(pol, dtype='int32')
    output[pol < -mu] = VADER_POLARITY_NEG
    output[pol > mu] = VADER_POLARITY_POS
    output[np.logical_and(pol >= -mu, pol <= mu)] = VADER_POLARITY_NEU
    return output

def compute_sentiment(sct_texts, quantize=True):
    polarity_train_begin = vader_polarity_scores(sct_texts.begin).reshape(-1, 4, 4)
    polarity_train_end_one = vader_polarity_scores(sct_texts.end_one)
    all_end_one = np.concatenate([polarity_train_begin, polarity_train_end_one[:, None, :]], axis=1)
    if quantize:
        all_end_one = vader_quantize_polarities(all_end_one)
    if sct_texts.end_two is None:
        return all_end_one, None
    else:
        polarity_train_end_two = vader_polarity_scores(sct_texts.end_two)
        all_end_two = np.concatenate([polarity_train_begin, polarity_train_end_two[:, None, :]], axis=1)
        if quantize:
            all_end_two = vader_quantize_polarities(all_end_two)
        return all_end_one, all_end_two
