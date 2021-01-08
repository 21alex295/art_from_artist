from src.featureselection import FeatureSelector
from src.imagereader import ImageReader
from src.randomforestclass import RandomForest
import multiprocessing as mp
import matplotlib.pyplot as plt
import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np

def extract_features():
    reader = ImageReader()
    images, labels = reader.open_images_small()
    ft = FeatureSelector()
    pool = mp.Pool(mp.cpu_count())
    ft = FeatureSelector()

    image_list = pool.map(ft.get_all, tqdm.tqdm(images))
    pickle.dump(image_list, open( "features.p", "wb" ))
    pickle.dump(labels, open( "labels.p", "wb" ))


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
        aux = np.concatenate((feats[i][3][:,:,0].flatten(), feats[i][3][:,:,1].flatten()))
        feats[i][3] = np.concatenate((aux, feats[i][3][:,:,2].flatten()))

        aux = np.concatenate((feats[i][0], feats[i][1]))
        aux = np.concatenate((aux, feats[i][2]))
        aux = np.concatenate((aux, feats[i][3]))
        feats[i] = np.concatenate((aux, feats[i][4]))
    return feats

def play():
    feats = pickle.load( open( "features.p", "rb" ) )
    labels = pickle.load( open( "labels.p", "rb" ) )

    skf = StratifiedKFold(n_splits=3)
    feats = flatten_features(feats)
    print(len(feats))
    print(len(feats[]))