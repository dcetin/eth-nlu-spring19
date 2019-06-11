from keras import layers, models, initializers, optimizers
import keras.backend as K
import numpy as np
import tensorflow as tf

def custom_loss(logits):
    def loss(y_true, y_pred):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(tf.squeeze(y_true), 'int32'),
            logits=logits)

        non_zero_weights = tf.cast(tf.sign(y_true), tf.float32)
        non_zero_counts = tf.cast(tf.count_nonzero(non_zero_weights, axis=1), tf.float32)
        loss_per_sentence = tf.reduce_sum(crossent * non_zero_weights, axis=1) / non_zero_counts

        return tf.reduce_mean(loss_per_sentence)
    return loss

# get glove coeff matrix
def get_glove_embeddings(fname, embedding_dim, word_index):
    embeddings_index = {}
    with open(fname, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    # prepare pre-learned embedding matrix
    num_words = len(word_index)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        #if i > VOCABULARY_SIZE:
        #    continue
        embedding_vector = embeddings_index.get(word)
        if not embedding_vector is None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(low=-0.25, high=0.25, size=embedding_dim)
    return embedding_matrix

# Default model
def default_fn(
        vocabulary,
        hidden_size=150,
        batch_size=50,
        max_seq_len=90,
        rnn_type='GRU',
        use_gpu=True,
        num_layers=3,
        learning_rate=0.001,
        dropout_rate=0.5,
        loss_weights=None,
        use_pre_trained_embeddings=True):

    if use_gpu:
        print('*** use rnn gpu implementation')
        rnn_layer = layers.CuDNNLSTM if rnn_type.upper() == 'LSTM' else layers.CuDNNGRU
    else:
        print('*** use rnn cpu implementation')
        rnn_layer = layers.LSTM if rnn_type.upper() == 'LSTM' else layers.GRU

    if use_pre_trained_embeddings:
        embeddings_initializer = initializers.Constant(
            get_glove_embeddings('data/glove.6B.100d.txt', 100, vocabulary))
        print('*** loaded glove embeddings ***')
    else:
        embeddings_initializer = 'uniform'
        print('*** use uniform embeddings ***')

    inputs = layers.Input((max_seq_len,), batch_shape=(batch_size, max_seq_len))
    x = layers.Embedding(len(vocabulary), hidden_size,
                               input_length=max_seq_len,
                               embeddings_initializer=embeddings_initializer)(inputs)
    for _ in range(num_layers):
        x = rnn_layer(hidden_size, return_sequences=True, stateful=True)(x)
        x = layers.Dropout(dropout_rate)(x)

    last_out = layers.Lambda(lambda x: x[:,-1])(x)

    polarities = layers.Dense(3, name='polarities')(last_out)
    polarities = layers.Activation('softmax')(polarities)

    logits = layers.TimeDistributed(layers.Dense(len(vocabulary)), name='logits')(x)
    lm = layers.Activation('softmax')(logits)

    optimizer = optimizers.Adam(lr=learning_rate, clipnorm=1.0)

    model = models.Model(inputs, [lm, polarities, last_out])
    model.compile(loss=[custom_loss(logits), 'sparse_categorical_crossentropy', None], optimizer=optimizer, loss_weights=loss_weights)
    model.summary()

    return model

from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

# Attention on sentiment
def attention_fn(
        vocabulary,
        hidden_size=150,
        batch_size=50,
        max_seq_len=90,
        rnn_type='GRU',
        use_gpu=True,
        num_layers=3,
        learning_rate=0.001,
        dropout_rate=0.5,
        loss_weights=None,
        use_pre_trained_embeddings=True):

    if use_gpu:
        print('*** use rnn gpu implementation')
        rnn_layer = layers.CuDNNLSTM if rnn_type.upper() == 'LSTM' else layers.CuDNNGRU
    else:
        print('*** use rnn cpu implementation')
        rnn_layer = layers.LSTM if rnn_type.upper() == 'LSTM' else layers.GRU

    if use_pre_trained_embeddings:
        embeddings_initializer = initializers.Constant(
            get_glove_embeddings('data/glove.6B.100d.txt', 100, vocabulary))
        print('*** loaded glove embeddings ***')
    else:
        embeddings_initializer = 'uniform'
        print('*** use uniform embeddings ***')

    inputs = layers.Input((max_seq_len,), batch_shape=(batch_size, max_seq_len))
    x = layers.Embedding(len(vocabulary), hidden_size,
                               input_length=max_seq_len,
                               embeddings_initializer=embeddings_initializer)(inputs)
    for _ in range(num_layers):
        x = rnn_layer(hidden_size, return_sequences=True, stateful=True)(x)
        x = layers.Dropout(dropout_rate)(x)

    # last_out = SeqSelfAttention(attention_activation='sigmoid')(x)
    last_out = SeqWeightedAttention()(x)
    # last_out = SeqSelfAttention(attention_activation='sigmoid', attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x)
    # last_out = layers.Lambda(lambda x: x[:,-1])(x)

    polarities = layers.Dense(3, name='polarities')(last_out)
    polarities = layers.Activation('softmax')(polarities)

    logits = layers.TimeDistributed(layers.Dense(len(vocabulary)), name='logits')(x)
    lm = layers.Activation('softmax')(logits)

    optimizer = optimizers.Adam(lr=learning_rate, clipnorm=1.0)

    model = models.Model(inputs, [lm, polarities, last_out])
    model.compile(loss=[custom_loss(logits), 'sparse_categorical_crossentropy', None], optimizer=optimizer, loss_weights=loss_weights)
    model.summary()

    return model

model_zoo = {'default_model': default_fn, 'attention_model': attention_fn, 'model-1': default_fn, 'model-2': attention_fn}