from src.featureselection import FeatureSelector
from src.imagereader import ImageReader
import multiprocessing as mp
import matplotlib.pyplot as plt
import tqdm


def play():

    reader = ImageReader()

    images, labels = reader.open_images_small()
    i = 0
    images_size = len(images)
    features = []
    print("Extracting features: ")

    pool = mp.Pool(mp.cpu_count())
    ft = FeatureSelector()

    image_list = pool.map(ft.get_all, images)
    pool.close()
