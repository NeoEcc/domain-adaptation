import h5py
import numpy as np
from glob import glob
import napari
import os
import torch_em
import time

import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import rescale



def get_dataloader(paths, data_key, label_key, split, patch_shape, batch_size = 1):
    """"
    Returns training and validation dataloaders 
    using `default_segmentation_loader` from torch_em
    Args:
        path: list of paths to the training / validation data
        data_key: key to access data in the files (such as "raw_crop" if inside an hdf5 file)
        label_key: key to access labels (such as "label_crop/mito")
        split: percentage of the data assigned to validation, from 0 to 1
        patch_shape: tuple representing the size of each patch
        batch_size: number of items used in every batch in training
    Returns:
        train_loader
        val_loader
    
    """

   
    # For boundary and foreground predictions
    label_transform = torch_em.transform.label.BoundaryTransform(
        add_binary_target=True
    )
    train_data_paths, val_data_paths = train_test_split(paths, test_size=split, random_state=42)
    # Case of the files containing both data and labels
    train_label_paths = train_data_paths
    val_label_paths = val_data_paths
    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size,
        label_transform=label_transform, label_transform2=None
    )
    # Define loaders
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_label_paths, label_key,
        rois=None, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(
        val_data_paths, data_key, val_label_paths, label_key,
        rois=None, **kwargs
    )
    
    return train_loader, val_loader

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