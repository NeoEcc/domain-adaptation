import h5py
import numpy as np
import os
import torch_em
import torch

from torch_em.data.sampler import MinInstanceSampler
from sklearn.model_selection import train_test_split
from shutil import copyfile


def check_inference(model, path_to_file, slice_shape = (128,)*3, path_to_raw = "raw_crop"):
    """
    Perform inference using a given model on data from an HDF5 file and save the results 
    to a new HDF5 file appending the inference results (e.g., "foreground" and "boundaries")
     to the new file.
    Args:
        model (torch.nn.Module): The PyTorch model to use for inference
        path_to_file (str): Path to the input HDF5 file.
        slice_shape (tuple of int, optional): The shape of the slices to be used for inference. 
            Defaults to (128, 128, 128).
        path_to_raw (str, optional): The key in the HDF5 file that contains the raw data 
            to be used for inference. Defaults to "raw_crop".
    """
    print("Checking inference for ", path_to_file)
    # Copy the file, do checks on the extension
    if ".h5" in path_to_file: 
        path_to_copy = path_to_file.replace(".h5", "_inference.h5")
    elif ".hdf5" in path_to_file:
        path_to_copy = path_to_file.replace(".hdf5", "_inference.hdf5")
    else:
        raise ValueError("Passed file must be HDF5 (.h5 or .hdf5), got " + path_to_file)
    copyfile(path_to_file, path_to_copy)
    
    # Prepare for inference and get data
    model.eval()
    loader = get_inference_dataloader([path_to_file], path_to_raw, slice_shape)
    prediction = []
    try:
        with torch.no_grad(): # Disable gradients
            for batch in loader:
                print("No problemo")
                data = batch[0]
                prediction.append(model(data))
                # Only one element
        # add data to copy of file
        with h5py.File(path_to_copy, "r+") as f:
            # print(f.keys())
            # Add a deletion of all files but mito and raw?
            for x in range(len(prediction)):
                f.create_dataset("foreground", slice_shape, np.uint8, prediction[x][0][0])
                f.create_dataset("boundaries", slice_shape, np.uint8, prediction[x][0][1])
    except Exception as e:
        print("Failed to test inference: ", e)
        if os.path.exists(path_to_copy):
            os.remove(path_to_copy)

def directory_to_path_list(directory) -> list:
    """
    Given the path to a certain directory, 
    returns the list of paths to all the files it contains.

    Args
        directory (str): path to a folder containing files. 
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
        train_data_paths, data_key, train_label_paths, label_key,
        rois = None, sampler = sampler, with_padding = True, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(
        val_data_paths, data_key, val_label_paths, label_key, #raw_transform=
        rois = None, sampler = sampler, with_padding= True,  **kwargs
    )
    
    return train_loader, val_loader

def get_inference_dataloader(paths, raw_key, patch_shape, batch_size = 1, num_workers = 1):
    """
    Returns a dataloader for inference, for now still with labels.
    Args:
        paths: list of paths to the inference data
        raw_key: key to access raw data in the files (such as "raw_crop" if inside an hdf5 file)
        patch_shape: tuple representing the size of each patch
        batch_size: number of items used in every batch
        num_workers: number of workers for data loading
    Returns:
        inference_loader
    """
    
    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size,
        label_transform=None, label_transform2=None,
        num_workers = num_workers
    )
    # Define loaders
    inference_loader = torch_em.default_segmentation_loader(
        paths, raw_key, paths, raw_key,
        rois = None, with_padding = True, **kwargs
    )
    return inference_loader

if __name__ == "__main__":
    test_dataloader = get_inference_dataloader(
        ["/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_3.h5"],
        "raw_crop",
        (128,)*3,
        )
    print(test_dataloader)