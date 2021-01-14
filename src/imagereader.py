"""image manager for the machine learning proyect."""

import os
import cv2
import multiprocessing as mp
import tqdm


class ImageReader():
    """Manages input and output of the image dataset."""

    def __init__(self):
        """Initialize class variables."""
        self._smalldata_path = "Dataset/final/"
        self._fulldata_path = "Dataset/images/"

    def _save_image(self, image_name):
        author = image_name.split("_")[0]
        image = cv2.imread(self._smalldata_path + image_name)
        if image is None:
            raise Exception("Path \"{}\" is not an image".format(
                self._smalldata_path + image_name))
        return (image, author)

    def open_images_small(self):
        """Open images of the resized dataset and saves them to a list."""
        image_list = []
        label_list = []
        i = 0
        im_list = os.listdir(self._smalldata_path)
        im_list_size = len(im_list)
        print("Loading images: ")
        pool = mp.Pool(mp.cpu_count())

        image_list = pool.map(self._save_image, tqdm.tqdm(im_list))
        pool.close()
        label_list = [elem[1] for elem in image_list]
        image_list = [elem[0] for elem in image_list]
        return image_list, label_list

    def open_sample(self):
        """Return single image for testing."""
        return cv2.imread(self._smalldata_path + "Andy_Warhol_99.jpg")
