
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from helper import get_fingerprints_batch
import numpy as np
from joblib import dump, load
import pickle
import logging
import time
from sklearn.metrics import roc_auc_score, classification_report

np.random.seed(20)

logger = logging.getLogger("svm_logger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s;%(message)s")
fh = logging.FileHandler("logs\\svm.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def svm():
    """Assumes model is an estimator class and search_grid is a dict, 
    in which the keys of the dict are the names of the parameters for
    the model we want to optimize and the values are the parameter 
    settings we want to try out."""
    
    clf = SVC(C=10, gamma=0.01, probability=True)
    
    X, y = get_fingerprints_batch()
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=None)
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
        metrics = svm()
        for k, v in zip(names, metrics):
            m[k] += v
    
    for k in names:
        m[k] = m[k]/N_ITERATIONS
        
    dump(m, 'svmmetrics.joblib')
