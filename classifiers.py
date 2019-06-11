import os
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import numpy as np

def run_classifiers(PREFIX):
    RANDOM_SEED = 42

    from numpy.random import seed
    seed(RANDOM_SEED)

    from tensorflow import set_random_seed
    set_random_seed(RANDOM_SEED)


    X_train_tasks = np.loadtxt(os.path.join(PREFIX, 'dev-transform-features.tsv'))
    X_train_hidden = np.load(os.path.join(PREFIX, 'dev-transform-hidden.npy'))
    y_train = np.loadtxt(os.path.join(PREFIX, 'dev-transform-labels.tsv'))

    X_test_tasks = np.loadtxt(os.path.join(PREFIX, 'test-transform-features.tsv'))
    X_test_hidden = np.load(os.path.join(PREFIX, 'test-transform-hidden.npy'))
    y_test = np.loadtxt(os.path.join(PREFIX, 'test-transform-labels.tsv'))

    X_report_tasks = np.loadtxt(os.path.join(PREFIX, 'report-transform-features.tsv'))
    X_report_hidden = np.load(os.path.join(PREFIX, 'report-transform-hidden.npy'))

    def build_model(n_hidden=1000, dropout=0.4):
        model = models.Sequential()
        model.add(layers.Dense(n_hidden))
        model.add(layers.Dropout(dropout))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(1))
        model.add(layers.Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def select_features(feature_type, X_tasks, X_hidden):
        if feature_type == 'proba+sentiment':
            return X_tasks
        elif feature_type == 'proba':
            return X_tasks[:, :6]
        elif feature_type == 'sentiment':
            return X_tasks[:, 6:]
        elif feature_type == 'hidden':
            return X_hidden
        elif feature_type == 'hidden+sentiment':
            return np.concatenate([X_hidden, X_tasks[:, 6:]], axis=1)
        elif feature_type == 'hidden+proba+sentiment':
            return np.concatenate([X_hidden, X_tasks], axis=1)
        else:
            raise NotImplemented()
            
    feature_types = ['proba', 'sentiment', 'hidden', 'proba+sentiment', 'hidden+sentiment', 'hidden+proba+sentiment']
    classifiers = [
        ('gnb', 'GaussianNB', lambda: GaussianNB()),
        ('svc', 'SVC(gamma=auto)', lambda: SVC(gamma='auto', random_state=RANDOM_SEED)),
        ('rfc', 'RandomForestClassifier(n_estimators=100)', lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
        ('nnc', 'NN(n_hidden=500, dropout=0.4)', lambda: KerasClassifier(build_model, epochs=5, batch_size=5, shuffle=True, n_hidden=500, verbose=0, dropout=0.4)),
        ('nnc', 'NN(n_hidden=1000, dropout=0.4)', lambda: KerasClassifier(build_model, epochs=5, batch_size=5, shuffle=True, n_hidden=1000, verbose=0, dropout=0.4))
    ]

    result = []
    for model_shortname, model_name, model_fn in classifiers:
        model = model_fn()
        for feature_type in feature_types:
            print('*** evaluating %s/%s...' % (model_name, feature_type))
            X_train = select_features(feature_type, X_train_tasks, X_train_hidden)
            X_test = select_features(feature_type, X_test_tasks, X_test_hidden)
            X_report = select_features(feature_type, X_report_tasks, X_report_hidden)
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_report_pred = model.predict(X_report)
            
            np.savetxt(os.path.join(PREFIX, 'classifier-predictions-%s-%s.tsv' % (model_shortname, feature_type)),
                       y_report_pred.squeeze(), fmt='%d')
            
            acc_train = accuracy_score(y_train, y_train_pred)
            acc_test = accuracy_score(y_test, y_test_pred)
            
            result.append({
                'classifier': model_name,
                'features': feature_type,
                'acc_train': acc_train,
                'acc_test': acc_test
            })
            
    df = pd.DataFrame.from_records(result, columns=['classifier', 'features', 'acc_train', 'acc_test'])
    df.to_csv(os.path.join(PREFIX, 'classifier-accuracies.tsv'), sep='\t')

    print(df)

    print(df.sort_values(by='acc_test', ascending=False))