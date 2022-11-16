"""Trying to find the right hyper-parameters for our ANN classifier"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from helper import get_fingerprints_batch, clock, recursive_feature_elimination
import logging
import numpy as np
from joblib import dump
import time
from sklearn.metrics import roc_auc_score, classification_report

np.random.seed(20)

logger = logging.getLogger('neural_network_debugging')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
fh = logging.FileHandler('logs\\mlp.log')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

logger.addHandler(fh)
#logger.addHandler(sh)


@clock
def mlp():
    X, y = get_fingerprints_batch()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=None)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(512,), alpha=0.00000003, max_iter=5_000,
        learning_rate='adaptive', activation='relu', random_state=42
        )
    
    clf.fit(X_train, y_train)
    t0 = time.time()
    y_pred = clf.predict(X_val)
    t1 = time.time()
    t = t1 - t0
    
    aur = roc_auc_score(y_val, y_pred)
    report = classification_report(y_true=y_val, y_pred=y_pred, target_names=('class 0', 'class 1'), output_dict=True)
    precision, recall, f1_score, _ = report['weighted avg'].values()
    
    return aur, report['accuracy'], precision, recall, f1_score, t


N_ITERATIONS = 100;
def main():
    names = ('area', 'accuracy', 'precision', 'recall', 'f1_score', 't')
    m = {key: 0 for key in names}

    for t in range(N_ITERATIONS):
        metrics = mlp()
        for k, v in zip(names, metrics):
            m[k] += v
    
    for k in names:
        m[k] = m[k]/N_ITERATIONS
        
    dump(m, 'mlpmetrics.joblib')
    