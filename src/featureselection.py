"""Extract features from the images."""
from skimage.feature import hog
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d


class FeatureSelector():
    """Feature selection object."""

    def __init__(self):
        """Initialize instance variables."""
        self._gaborfilters = self._gaborbank()  # might want to externalize

    def hogdescriptor(self, image):
        """Get hog descriptor from image."""
        fd, hog_image = hog(np.array(image),
                            orientations=8,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2),
                            visualize=True,
                            multichannel=True)
        hog_image = np.round(hog_image)
        n_bins = 200
        hog_hist, bins = np.histogram(hog_image.ravel(), n_bins, [0, n_bins])
        return hog_hist

    def global_sift(self, image):
        """Get the global sift from a image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(10)
        kp = sift.detect(gray, None)
        kp, des = sift.compute(gray, kp)
        return des

    def dense_sift(self, image):
        """Get sift in every 16x16 image cell."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(10)
        size_x, size_y = image.shape[0], image.shape[1]
        descriptors = []
        for x in range(0, size_x, 64):
            row = []
            for y in range(0, size_y, 64):
                kp = sift.detect(gray)
                kp, des = sift.compute(gray, kp)
                row.append(des)
            descriptors.append(row)
        return descriptors

    def lbp(self, image, n_points=8, radius=1, method="uniform"):
        """Get the local binary pattern from the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method)
        n_bins = int(lbp.max() + 1)
        lbp_hist, bins = np.histogram(lbp.ravel(), n_bins, [0, n_bins])
        return lbp

    def tinyimage(self, image):
        """Get the image reduced."""
        resized = cv2.resize(image,
                             (64, 64),
                             interpolation=cv2.INTER_AREA)
        return resized

    def lines(self, image):
        """Get the hough lines from the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        return lines

    def _gaborbank(self, freqrange=[0.1, 0.4]):
        theta = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4,
                 5*np.pi/6]
        filter_bank = []
        for t in theta:
            for freq in np.arange(freqrange[0], freqrange[1], 0.1):
                g = gabor_kernel(frequency=freq, theta=t)
                filter_bank.append(g)
        return filter_bank

    def _gist_cells(self, image, n_cells=4):
        size_x, size_y = image.shape[0], image.shape[1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cells = []
        cellsize_x = int(size_x/n_cells)
        cellsize_y = int(size_y/n_cells)
        for x in range(n_cells):
            for y in range(n_cells):
                cell = gray[x*cellsize_x:x*cellsize_x + cellsize_x,
                            y*cellsize_y:y*cellsize_y + cellsize_y]
                cells.append(cell)
        return cells

    def gist(self, image):
        """Implement the GIST descriptor."""
        cells = self._gist_cells(image)
        descriptor = []
        for cell in cells:
            for filter in self._gaborfilters:
                filtered = convolve2d(cell, filter, mode="same")
                cell_average = np.mean(filtered)
                descriptor.append(cell_average)
        return descriptor

    def color_histogram(self, image):
        """Get histograms from image."""
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        return hist_b, hist_g, hist_r

    def texton_histogram(self, image):
        """Get histogram texton from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = []
        for filter in self._gaborfilters:
            filtered = convolve2d(gray,
                                  filter,
                                  mode="same",
                                  boundary="wrap")
            filtered_image.append(np.real(filtered))
        texton_map = sum(filtered_image)
        # texton_hist = cv2.calcHist([texton_map], [0], None, [256], [0, 256])
        texton_hist, bins = np.histogram(texton_map.ravel(), 256, [0, 256])

        return texton_hist

    def get_all(self, image):
        """Extract all features from image."""
        return [  # self.gist(image),
            self.hogdescriptor(image),
            self.color_histogram(image),
            self.texton_histogram(image),
            self.tinyimage(image),
            self.lbp(image)]
