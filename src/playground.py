from src.featureselection import FeatureSelector
from src.imagereader import ImageReader
from src.randomforestclass import RandomForest
import multiprocessing as mp
import matplotlib.pyplot as plt
import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import seaborn as sns


def extract_features():
    reader = ImageReader()
    images, labels = reader.open_images_small()
    ft = FeatureSelector()
    pool = mp.Pool(mp.cpu_count())
    ft = FeatureSelector()
    image_list = pool.map(ft.hogdescriptor, tqdm.tqdm(images))
    #pickle.dump(image_list, open("features.p", "wb"))
    #pickle.dump(labels, open("labels.p", "wb"))


def flatten_features(feats):
    for i in range(len(feats)):
        # gist
        feats[i][0] = np.real(feats[i][0])
        # hog
        feats[i][1] = feats[i][1].flatten()

        # color hists
        aux = np.concatenate((feats[i][2][0], feats[i][2][1]))
        feats[i][2] = np.concatenate((aux, feats[i][2][2]))
        feats[i][2] = feats[i][2].reshape(1, -1)[0]
        # tiny images
        aux = np.concatenate((feats[i][3][:, :, 0].flatten(), feats[i][3][:, :, 1].flatten()))
        feats[i][3] = np.concatenate((aux, feats[i][3][:, :, 2].flatten()))

        aux = np.concatenate((feats[i][0], feats[i][2]))
        aux = np.concatenate((aux, feats[i][3]))
        feats[i] = np.concatenate((aux, feats[i][4]))
        # feats[i] = np.concatenate((aux, feats[i][4]))
    return feats


def string_to_int(labels):
    unique_labels = list(set(labels))
    n_labels = len(unique_labels)
    key_label = {}
    for i in range(n_labels):
        key_label[unique_labels[i]] = i
    for j in range(len(labels)):
        labels[j] = key_label[labels[j]]
    return labels


def play():
    extract_features()
    feats = pickle.load(open("Dataset/features.p", "rb"))
    labels = pickle.load(open("Dataset/labels.p", "rb"))
    unique_labels = set(labels)
    numeric_labels = string_to_int(labels)
    n_splits = 7
    skf = StratifiedKFold(n_splits=n_splits)
    feats = flatten_features(feats)
    feats = np.array(feats)
    numeric_labels = np.array(numeric_labels)
    acc = 0
    prec = 0
    cm = np.zeros((len(unique_labels), len(unique_labels)))
    for train_index, test_index in skf.split(feats, numeric_labels):
        X_train, X_test = feats[train_index], feats[test_index]
        y_train, y_test = numeric_labels[train_index], numeric_labels[test_index]
        rfc = RandomForest()
        rfc.train(X_train, y_train)
        rfc.predict(X_test, y_test)
        print(rfc.accuracy())
        acc += rfc.accuracy()
        prec += rfc.precision()
        cm += rfc.conf_matrix()
    accuracy = acc/n_splits
    precision = prec/n_splits
    print(accuracy)
    print(precision)
    print(cm)
    sns.heatmap(cm, annot=True, xticklabels=unique_labels, yticklabels=unique_labels)
    plt.show()
