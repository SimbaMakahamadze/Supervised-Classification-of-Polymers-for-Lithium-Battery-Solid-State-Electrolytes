import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

DATABASE_SIZE = 1_000_000


def clock(f):
    def wrapper(*args, **kwargs):
        import time
        t0 = time.time()
        r = f(*args, **kwargs)
        print(f"Time elapsed, {(time.time() - t0)/60: .0f} minutes")
        return r
    return wrapper


def get_fingerprint_from(smiles_string, fp_size=512):
    mol = Chem.MolFromSmiles(smiles_string)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=fp_size)
    array = np.zeros((1, ))
    DataStructs.ConvertToNumpyArray(fp, array)
    return array.reshape((1, -1))


def make_prediction_on(fp, clf, threshold=0.9):
    """Assumes fp (fingerprint) is an 1D-array, clf is an estimator and
    threshold is some value equal or greater than 0.5.
    Returns True if the prediction (probability) of clf on fp meets or
    exceeds the threshold.
    """
    assert threshold >= 0.5, f"Invalid threhold value={threshold}"
    return clf.predict_proba(fp.reshape(1, -1))[0][1] >= threshold


def get_recommendations_from(classifier, threshold):
    """This function is meant merely for testing and sanity checking purposes.
    Assumes classifier is a joblib filename storing an estimator with a 
    predict_proba method and threhold is a float between 0 and 1. Prints 
    useful information on the predictions that the classifier makes.
    """
    predictions = list()
    clf = load(classifier)
    with open("testing_PI1M.csv", 'r') as fin:
        for line in fin:
            index, smiles = line[:-1].split(',')
            fp = get_fingerprint_from(smiles.strip())
            predictions.append(make_prediction_on(fp, clf, threshold))
    print(f"Threshold = {threshold}")
    print(f"Number of positive labels = {sum(predictions)}")

def get_fingerprints_batch(fp_size=512):
    """Assumes fp_size is an int, the size or length of the fingerprint vector;
    filename is a csv file with two columns, a SMILES column, and Y column for
    corresponding labels.
    Calculates and returns the molecular fingerprint from SMILES string and 
    appends the corresponding label to the end of the vector. This vector is
    then appended to a list with the rest of the calculated fingerprints.
    """
    df = pd.read_csv('ml_ready.csv')
    molecules = [Chem.MolFromSmiles(s) for s in df['SMILES']]
    
    fingerprints = []
    for mol, label in zip(molecules, df['Y']):
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=fp_size)
        array = np.zeros((1, ))
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        sample = array.tolist()
        sample.append(label)
        fingerprints.append(sample)
    fp_and_label_matrix = np.array(fingerprints)
    X, y = fp_and_label_matrix[:, :-1], fp_and_label_matrix[:, -1]
    return X, y

def recursive_feature_elimination(estimator, X, y):
    rfecv = RFECV(estimator=estimator, step=1, scoring='accuracy', cv=StratifiedKFold(10))
    df_features = pd.DataFrame(columns = ['feature', 'support', 'ranking'])
    rfecv.fit(X, y)
    
    for i in range(X.shape[1]):
        row = {'feature': i,'support': rfecv.support_[i], 'ranking': rfecv.ranking_[i]}
        df_features = df_features.append(row, ignore_index=True)
    df_features.sort_values(by='ranking')
    df_features.to_csv('forest_feature_importance.csv')
    return rfecv

def evaluate(model, iterations=30):
    """Calculates and returns the average performance of an estimator
    specifically mlp from mlp.py and knn from knn.py. This function
    is meant for internal testing purposes only"""
    performance = [model() for _ in range(iterations)]
    return np.mean(performance)


def clock_factory(logger):
    def clock(model):
        def wrapper():
            r = model()
            logger.debug(f"score = {r}")
            return r
        return wrapper
    return clock


def calculate_roc_area(clf):
    X, y = get_fingerprints_batch()
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    pred = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred)

def get_recommendations_from_file(filename):
    """Returns the contents of the file as elements of a set"""
    with open(filename+".log") as fin:
        return {(line.rstrip("\n").split(";")[1]) for line in fin}

