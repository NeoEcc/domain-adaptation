import h5py
import numpy as np
import os
import torch_em
import torch

from torch_em.data.sampler import MinInstanceSampler
from torch_em.util import prediction
from torch_em.loss.dice import dice_score
from sklearn.model_selection import train_test_split
from shutil import copyfile
from torchmetrics import JaccardIndex 
from skimage import morphology
from scipy.ndimage import label
import napari_segment_blobs_and_things_with_membranes as nsbatwm



def directory_to_path_list(directory) -> list:
    """
    Given the path to a certain directory, 
    returns the list of paths to all the files it contains.

    Args
        directory (str): path to a folder containing files. 
    Returns:
        paths_list
    """
    # Input check: add a "/" if not there not to mess up later
    if directory == "":
        raise ValueError("Path cannot be empty")
    if directory[-1] != "/":
        directory += "/"
    all_paths_raw = os.listdir(directory)
    all_paths = []
    for i in range(len(all_paths_raw)):
        all_paths.append(f"{directory}{all_paths_raw[i]}")
    return all_paths

def get_dataloader(paths, data_key, label_key, split, patch_shape, batch_size = 1, num_workers = 1):
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

    sampler = MinInstanceSampler(2, 0.95, min_size = 2500)
   
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
        label_transform=label_transform, label_transform2=None,
        num_workers = num_workers
    )
    # Define loaders
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_label_paths, label_key, raw_transform=minmax_norm,
        rois = None, sampler = sampler, with_padding = True, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(
        val_data_paths, data_key, val_label_paths, label_key, raw_transform=minmax_norm,
        rois = None, sampler = sampler, with_padding= True,  **kwargs
    )
    
    return train_loader, val_loader

def minmax_norm(x):
    """
    MinMax normalization for each instance of the sample
    """
    min = 0
    max = 255
    x = x.astype(np.float32)
    # return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x - min)/(max - min) 

if __name__ == "__main__":
    test_folder = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_inference"
    # Test the post processing with some new files from old files
    # Calculate IOU and dice before and after!!
