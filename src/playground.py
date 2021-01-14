from src.featureselection import FeatureSelector
from src.imagereader import ImageReader
from src.classifier import Classifier
from src.feature_processor import FeatureProcessor
import src.print_utils as pru
import multiprocessing as mp
import matplotlib.pyplot as plt
import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


# FeatureProcesor.extract_features()

def load_data():
    feats = pickle.load(open("Dataset/features.p", "rb"))
    labels = pickle.load(open("Dataset/labels.p", "rb"))
    unique_labels = set(labels)
    fp = FeatureProcessor()
    numeric_labels = fp.string_to_int(labels)
    feats = fp.flatten_features(feats)
    feats = np.array(feats)
    numeric_labels = np.array(numeric_labels)
    return feats, numeric_labels, unique_labels


def traintestsplit(feats, numeric_labels):
    X_train, X_test, y_train, y_test = train_test_split(feats, 
                                                        numeric_labels, 
                                                        test_size=0.2, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def simple_classifier(X_train, X_test, y_train, y_test,classifier="rf", params=[500  , None]):
    rfc = Classifier(classifier, params)
    rfc.train(X_train, y_train)
    rfc.predict(X_test, y_test)
    print("--accuracy: {}, precision: {}".format(rfc.accuracy(), rfc.precision()))
    return rfc.accuracy(), rfc.precision(), rfc.conf_matrix()


def parameter_selector(algorithm):
        params = []
        if algorithm == "rf":
            params = [[10, None]]
        elif algorithm == "svc":
                    params = [[1, "linear", 1]]
        else:
            raise ValueError("algorithm is not defined, please select SVC or RF")
        return params

def hyperparametrization(feats, numeric_labels, unique_labels, n_splits=10):
    results = {}
    skf = StratifiedKFold(n_splits=n_splits)

    for algorithm in ["rf", "svc"]:
        params = parameter_selector(algorithm)
        for param in params:
            acc_error_svc = []
            prec_error = []
            precision = 0
            accuracy = 0
            acc = 0
            prec = 0
            cm = np.zeros((len(unique_labels), len(unique_labels)))
            for train_index, test_index in skf.split(feats, numeric_labels):
                X_train, X_test = feats[train_index], feats[test_index]
                y_train, y_test = numeric_labels[train_index], numeric_labels[test_index]
                #rfc = Classifier("rf", [50, None])
                local_acc, local_prec, local_cm = simple_classifier(X_train, 
                                                                    X_test, 
                                                                    y_train, 
                                                                    y_test, 
                                                                    algorithm, 
                                                                    param)
                acc += local_acc
                prec += local_prec
                cm += local_cm
            accuracy = acc / n_splits
            precision = prec / n_splits
            results[str(param)] = [accuracy, precision]
    print(results)
    return results


"""

"""

def play():
    feats, numeric_labels, unique_labels = load_data()
    results = hyperparametrization(feats, numeric_labels, unique_labels, n_splits=10)
    X_train, X_test, y_train, y_test = traintestsplit(feats, numeric_labels)
    accuracy, precision, cm = simple_classifier(X_train, X_test, y_train, y_test, "rf", [500, None])
    pru.print_cm(cm, unique_labels)

    accuracy_bar = [52.96, 64.35, 83.66, 67.82, 59.40, 71.48]
    precision_bar = [54.95, 62.59, 84.49, 67.64, 62.19, 70.48]

    pru.print_bars(accuracy_bar, precision_bar)
    pru.print_metrics(accuracy_bar, precision_bar)