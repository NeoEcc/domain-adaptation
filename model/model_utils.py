import h5py
import numpy as np
from glob import glob
import napari
import os

import torch.utils.data.dataset
import torch_em
import torch


from torch_em.data.sampler import MinInstanceSampler
import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from shutil import copyfile



def check_inference(model, path_to_file, slice_shape = (128,)*3, path_to_raw = "raw_crop"):
    """
    Given a model and a path to an HDF5 file, copies the file as x_inference.h5 
    and adds to thenew  file the inference produced by the model.
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
    
    # Prepare for inference and open data
    model.eval()
    try:
        with h5py.File(path_to_copy, "r+") as f:
            data = f[path_to_raw][:]

            # Dimensionality check
            if len(data.shape) == 3:
                # data.unsqueeze(0)
                data = np.expand_dims(data, axis=0)  
                data = np.expand_dims(data, axis=0)  # Shape becomes (1, 1, D, H, W)
                
            elif len(data.shape) == 4:
                data = np.expand_dims(data, axis=0)  
            elif len != 5:
                raise ValueError(f"Unexpected data shape: {data.shape}, expected (1, 1, D, H, W)")

            print("Initial: ", data.shape, " type: ", type(data))
            # Use negative index to handle all data shapes

            if(slice_shape[-1] > data.shape[-1]) or (slice_shape[-3] > data.shape[-3]):
                add_shape = (
                    (0,)*2, 
                    (0,)*2,
                    (int((slice_shape[-3]-data.shape[-3])/2),)*2, 
                    (int((slice_shape[-2]-data.shape[-2])/2),)*2, 
                    (int((slice_shape[-1]-data.shape[-1])/2),)*2, 
                    )
                data = np.pad(data, add_shape, mode = "constant", constant_values= 0)

            if data.shape[-3] > slice_shape [-3] or data.shape[-1] > slice_shape [-1]:
                data = resize(data, (1, 1) + slice_shape, anti_aliasing= True)
            data_tensor = torch.from_numpy(data).float()  # Convert to PyTorch tensor
            result = model(data_tensor)
            # Suppose we get (1, 2, z, y, x)
            # print("Result: ", result[0][0])

            # TODO: implement same thing with dataloader instead1
            f.create_dataset("foreground", result[0][0].shape,  np.float32, )
            f.create_dataset("boundaries", result[0][1].shape,  np.float32, result[0][1].detach().numpy())
            # f["foreground"] = (result[0][0].detach().numpy() * 255).astype(np.uint8)
            # f["boundaries"] = (result[0][1].detach().numpy() * 255).astype(np.uint8)
    except Exception as e:
        print(f"Failed test inference for {path_to_copy}: ", e)
        if os.path.exists(path_to_copy):
            os.remove(path_to_copy)
        raise e

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
        val_data_paths, data_key, val_label_paths, label_key,
        rois = None, sampler = sampler, with_padding= True,  **kwargs
    )
    
    return train_loader, val_loader

def get_inference_dataloader(paths, raw_key, patch_shape, batch_size = 1, num_workers = 1):
    # TODO!
    """
    Returns a dataloader for inference without labels.
    Args:
        paths: list of paths to the inference data
        raw_key: key to access raw data in the files (such as "raw_crop" if inside an hdf5 file)
        patch_shape: tuple representing the size of each patch
        batch_size: number of items used in every batch
        num_workers: number of workers for data loading
    Returns:
        inference_loader
    """
    dataset = torch.utils.data.IterableDataset(
        paths=paths,
        raw_key=raw_key,
        patch_shape=patch_shape,
        ndim=3
    )
    inference_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return inference_loader



def get_random_colors(labels):
    n_labels = len(np.unique(labels)) - 1
    cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap

def plot_samples(image, labels, cmap="gray", view_napari=True):
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

if __name__ == "__main__":
    test_dataloader = get_inference_dataloader(
        ["/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_3.h5"],
        "raw_crop",
        (128,)*3,
        )
    print(test_dataloader)