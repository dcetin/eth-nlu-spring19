from sct_dataset import read_sct_stories, sct_stories_to_sequences
from tokenizer import Tokenizer
from sentiment import compute_sentiment
import numpy as np
import os
import pickle

DATA_DIR = 'data'
CACHE_DIR = 'cache'
VOCABULARY_SIZE = 20000
MAX_SEQ_LEN = 91
VOCABULARY_FILE = 'sct_train.csv'

class DataLoader:
    def __init__(self, data_dir=DATA_DIR, cache_dir=CACHE_DIR, max_vocabulary_size=VOCABULARY_SIZE, max_seq_len=MAX_SEQ_LEN):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.max_vocabulary_size = max_vocabulary_size
        self.max_seq_len = max_seq_len
        # create cache dir if it does not exist
        os.makedirs(cache_dir, exist_ok=True)

    def get_tokenizer(self):
        tokenizer_fname = os.path.join(self.cache_dir,
            'tokenizer_maxvocab-%d.pickle' % self.max_vocabulary_size)
        try:
            with open(tokenizer_fname, 'rb') as f:
                tok = pickle.load(f)
        except:
            texts_train = read_sct_stories(os.path.join(self.data_dir, VOCABULARY_FILE))
            tok = Tokenizer(max_vocabulary_size=VOCABULARY_SIZE).fit(
                texts_train.begin + texts_train.end_one)
            with open(tokenizer_fname, 'wb') as f:
                pickle.dump(tok, f)
        return tok

    def get_data(self, name):
        tok = self.get_tokenizer()
        cache_fname = os.path.join(self.cache_dir, 'data_%s_maxseqlen-%d.pickle' % (name, self.max_seq_len))
        data_fname = os.path.join(self.data_dir, name)
        try:
            with open(cache_fname, 'rb') as f:
                data = pickle.load(f)
        except:
            data = {}
            stories = read_sct_stories(data_fname)
            sequences = sct_stories_to_sequences(tok.texts_to_sequences, stories, max_seq_len=self.max_seq_len)
            data['stories_one'] = np.concatenate([sequences.begin, sequences.end_one[:, None, :]], axis=1)
            if not sequences.end_two is None:
                data['stories_two'] = np.concatenate([sequences.begin, sequences.end_two[:, None, :]], axis=1)
                data['stories_correct'] = sequences.correct_end
            sentiment_one, sentiment_two = compute_sentiment(stories)
            data['sentiment_one'] = sentiment_one
            if not sentiment_two is None:
                data['sentiment_two'] = sentiment_two
            with open(cache_fname, 'wb') as f:
                pickle.dump(data, f)
        return data
