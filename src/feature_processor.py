from src.featureselection import FeatureSelector
from src.imagereader import ImageReader
from src.classifier import Classifier
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

class FeatureProcessor:

    def extract_features(self):
        reader = ImageReader()
        images, labels = reader.open_images_small()
        ft = FeatureSelector()
        pool = mp.Pool(mp.cpu_count())
        ft = FeatureSelector()
        for image in images:
            ft.get_all(image)
        #image_list = pool.map(ft.get_all, tqdm.tqdm(images))
        #pickle.dump(image_list, open("Dataset/features.p", "wb"))
        #pickle.dump(labels, open("Dataset/labels.p", "wb"))

    def _correct_gist(self, feat):
        return np.real(feat)

    def _correct_hog(self, feat):
            return [item for sublist in feat for item in sublist]

    def _correct_color(self, feat):
        aux = np.concatenate((feat[0], feat[1]))
        feat = np.concatenate((aux, feat[2]))
        feat = feat.reshape(1, -1)[0]
        return feat

    def _correct_tiny(self, feat):
        aux = np.concatenate((feat[:, :, 0].flatten(), feat[:, :, 1].flatten()))
        feat = np.concatenate((aux, feat[:, :, 2].flatten()))
        return feat

    def _correct_lbp(self, feat):
        n_bins = int(feat.max() + 1)
        lbp_hist, bins = np.histogram(feat.ravel(), n_bins, [0, n_bins])
        return lbp_hist

    def flatten_features(self, feats):
        for i in range(len(feats)):
            feats[i][0] =self._correct_gist(feats[i][0])
            feats[i][1] = self._correct_hog(feats[i][1])
            feats[i][2] = self._correct_color(feats[i][2])
            feats[i][4] = self._correct_tiny(feats[i][4])
            feats[i][5] = self._correct_lbp(feats[i][5])
            aux = np.concatenate((feats[i][0], feats[i][1]))
            aux = np.concatenate((aux, feats[i][2]))
            aux = np.concatenate((aux, feats[i][3]))
            aux = np.concatenate((aux, feats[i][4]))
            feats[i] = np.concatenate((aux, feats[i][5]))
        return feats


    def string_to_int(self, labels):
        unique_labels = list(set(labels))
        print(unique_labels)
        n_labels = len(unique_labels)
        key_label = {}
        for i in range(n_labels):
            key_label[unique_labels[i]] = i
        for j in range(len(labels)):
            labels[j] = key_label[labels[j]]
        return labels