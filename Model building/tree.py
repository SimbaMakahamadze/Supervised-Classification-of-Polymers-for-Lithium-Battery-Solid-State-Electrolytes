
"""Trying out different classifiers on our machine learning ready samples"""

from sklearn.tree import DecisionTreeClassifier
from helper import get_fingerprints_batch
from helper import clock_factory
import logging
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import pickle
from joblib import dump, load

np.random.seed(20)

logger = logging.getLogger('tree')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt="%(name)s; %(levelname)s: %(message)s")
fh = logging.FileHandler('logs\\tree.log', mode='a')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


@clock_factory(logger)
def tree():
    X, y = get_fingerprints_batch()
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=None)
    
    clf = DecisionTreeClassifier(max_features='auto', ccp_alpha=0.01, max_depth=9, criterion='gini')
   
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
        metrics = tree()
        for k, v in zip(names, metrics):
            m[k] += v
    
    for k in names:
        m[k] = m[k]/N_ITERATIONS
        
    dump(m, 'treemetrics.joblib')