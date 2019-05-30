import pickle
import os
from util import load_or_compute
from tokenizer import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
import csv

SCTStories = namedtuple('SCTStories', ('begin', 'end_real', 'end_fake'))
SCTSequences = namedtuple('SCTSequences', ('begin', 'end_real', 'end_fake'))

def read_sct_stories(fname, skip_header=True):
    beginnings = list()
    real_endings = list()
    fake_endings = list()
    with open(fname) as f:
        csvreader = csv.reader(f, delimiter=',')
        if skip_header:
            next(csvreader, None) # skip header
        for row in csvreader:
            if len(row) == 7: #ROCstories, only real endings
                beginnings.extend(row[2:-1])
                real_endings.append(row[-1])
            elif len(row) == 8: #Eval/test set, real and fake endings
                beginnings.extend(row[1:-3])
                realid = int(row[-1])
                if realid == 1:
                    real_endings.append(row[-3])
                    fake_endings.append(row[-2])
                else:
                    real_endings.append(row[-2])
                    fake_endings.append(row[-3])
            else:
                raise Exception('wrong number of items in input file')
    return SCTStories(beginnings, real_endings, fake_endings if len(fake_endings) > 0 else None)

def sct_stories_to_sequences(texts_to_sequences_func, sct_stories, max_seq_len=91):
    seq_b = pad_sequences(texts_to_sequences_func(sct_stories.begin), maxlen=max_seq_len)
    seq_b = seq_b.reshape(seq_b.shape[0]//4, 4, seq_b.shape[1])
    seq_r = pad_sequences(texts_to_sequences_func(sct_stories.end_real), maxlen=max_seq_len)
    seq_f = None
    if sct_stories.end_fake:
        seq_f = pad_sequences(texts_to_sequences_func(sct_stories.end_fake), maxlen=max_seq_len)
    return SCTSequences(seq_b, seq_r, seq_f)

class SCTCachedReader:
    TOKENIZER_FILE = 'tokenizer.pickle'

    def __init__(self, dirname, tokenizer, max_seq_len=91):
        if not os.path.isdir(dirname):
            raise ValueError('given cache directory does not exist')
        self.cachedir = dirname
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def read_stories(self, filename):
        def f():
            stories = read_sct_stories(filename)
            return sct_stories_to_sequences(self.tokenizer.texts_to_sequences, stories, self.max_seq_len)
        cache_fname = os.path.join(self.cachedir, os.path.basename(filename) + '.pickle')
        return load_or_compute(cache_fname, f)

    def sequences_to_texts(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)
