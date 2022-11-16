import pandas as pd
import logging
from joblib import load
from helper import get_fingerprint_from, make_prediction_on, clock


def setup_logging_for(name_of_estimator):
    """Instantiates and returns a logger used to log the index of
    entries that receive a positive label from the given estimator
    """
    logger = logging.getLogger(f"{name_of_estimator}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"logs\\{name_of_estimator}_recommendations.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s;%(message)s"))
    logger.addHandler(fh)
    return logger

@clock
def high_throughput_screen(name_of_estimator, decision_threshold=.999, db="PI1M.csv"):
    """Assumes db is text file with a SMILES column, name_of_estimator is a 
    classifier (knn, svm, nn, tree, forest) used to classify the polymers and 
    decision_threshold is a float.
    
    Writes the polymers classified as positive to an file."""
    logger = setup_logging_for(name_of_estimator)
    df = pd.read_csv(db)
    clf = load(name_of_estimator+".joblib")
    print(f"Screening of {db} using {name_of_estimator} in progress ...")
    for n, SMILES in enumerate(df["SMILES"].values.tolist()):
        fp = get_fingerprint_from(SMILES)
        if make_prediction_on(fp, clf, decision_threshold):
            logger.debug("%d" % n)
        print(n, file=open("logs\\progress_tracker.log", "w"))
        if (n%100_000 == 0):
            print("%", end=", ")