import h5py
import numpy as np
from glob import glob
import napari
import os

import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt

from torch_em.data import datasets

def get_random_colors(labels):
    n_labels = len(np.unique(labels)) - 1
    cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap

def plot_samples(image, labels, cmap="gray", view_napari=False):
    def _get_mpl_plots(image, labels):
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image, cmap=cmap)
        ax[0].axis("off")
        ax[0].set_title("Image")

        ax[1].imshow(labels, cmap=get_random_colors(labels), interpolation="nearest")
        ax[1].axis("off")
        ax[1].set_title("Labels")
        plt.show()

    if view_napari:
    
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        napari.run()
    else:
        _get_mpl_plots(image, labels)