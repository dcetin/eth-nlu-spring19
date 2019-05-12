from collections import Counter

class Tokenizer:
    """ Tokenizer is used to turn tokens (words and special tokens) into sequences of integers
        and vice versa.
    """
    TOKEN_PAD = 0
    TOKEN_BOS = 1
    TOKEN_EOS = 2
    TOKEN_UNK = 3
    
    def __init__(self, max_vocabulary_size=None):
        self.max_vocabulary_size = max_vocabulary_size
        
    def fit(self, texts_or_filename):
        def inner(texts):
            self.word_index = dict()
            self.index_word = dict()
            special_tokens = [
                ('<pad>', self.TOKEN_PAD),
                ('<bos>', self.TOKEN_BOS),
                ('<eos>', self.TOKEN_EOS),
                ('<unk>', self.TOKEN_UNK)
            ]

            counter = Counter(word.lower() for sent in texts for word in sent.split(' '))
            for word, index in special_tokens: 
                self.word_index[word] = index
                self.index_word[index] = word

            max_vocabulary_size = self.max_vocabulary_size
            if not self.max_vocabulary_size is None:
                max_vocabulary_size = self.max_vocabulary_size - 4

            offset = len(self.word_index)
            for index, (word, _) in enumerate(counter.most_common(max_vocabulary_size)):
                self.word_index[word] = index+offset
                self.index_word[index+offset] = word
            self.vocabulary_size = len(self.word_index)
        try:
            with open(texts_or_filename, 'r') as f:
                inner(line.strip() for line in f)
        except:
            inner(texts_or_filename)
        return self
    
    def texts_to_sequences_generator(self, texts, prepend_bos=True, append_eos=True):
        for text in texts:
            words = [word.lower() for word in text.split(' ')]
            sequence = list()
            if prepend_bos:
                sequence.append(1)
            for word in words:
                try:
                    sequence.append(self.word_index[word])
                except KeyError:
                    sequence.append(self.word_index['<unk>'])
            if append_eos:
                sequence.append(2)
            yield sequence
            
    def texts_to_sequences(self, texts, prepend_bos=True, append_eos=True):
        return list(self.texts_to_sequences_generator(texts, prepend_bos, append_eos))
    
    def sequences_to_texts_generator(self, sequences, suppress_padding=True):
        for seq in sequences:
            words = (self.index_word[idx] for idx in seq)
            if suppress_padding:
                words = (w for w in words if w != '<pad>')
            yield ' '.join(words)
            
    def sequences_to_texts(self, sequences, suppress_padding=True):
        return list(self.sequences_to_texts_generator(sequences, suppress_padding))
