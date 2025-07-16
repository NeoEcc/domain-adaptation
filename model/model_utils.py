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


def postprocess_prediction(foreground, boundaries = None, threshold=0.8, instance_separation = True):
    """
    Postprocess the prediction by thresholding and optionally separating instances.
    Args:
        prediction (torch.Tensor): The model's prediction tensor.
        threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        instance_separation (bool, optional): Whether to separate instances. Defaults to False.
    Returns:
        torch.Tensor: The postprocessed prediction tensor.
    """
    boundaries = None
    # Binarize the prediction
    prediction = (foreground > threshold)
    prediction = morphology.remove_small_holes(prediction, 2048) # 5000 in synapse
    prediction = morphology.remove_small_objects(prediction, 256)
    if boundaries is not None and instance_separation:
        # TODO: Remove boundaries, later add them back and expand the 
        # labels for ideal instance segmentation
        # TODO: evaluate if this is worth doing or not
        boundaries = (boundaries > threshold)
        diff = torch.logical_and(foreground, torch.logical_not(boundaries))
        prediction = nsbatwm.label(diff, connectivity=1)
        
    prediction = morphology.binary_opening(prediction, None)

    if instance_separation:
        # Separate and label
        # prediction = nsbatwm.split_touching_objects(prediction, sigma = 50)
        prediction_bool = prediction.astype(bool)
        prediction, num = label(prediction_bool)
        print("Instances separated: ", num) 
    
    return prediction.astype(float)
    

def check_inference(model, path_to_file, path_to_dest, original_shape = (128,)*3, block_size = (96,)*3, halo = (16,)*3, 
                    raw_key = "raw_crop", test_function = None, label_key = "labels/mito", postprocess = False):
    """
    Perform inference using a given model on data from an HDF5 file and save the results 
    to a new HDF5 file appending the inference results ("foreground" and "boundaries")
    to the new file.
    If the test function and label key is given, it returns the test metric.
    Args:
        model (torch.nn.Module): The PyTorch model to use for inference
        path_to_file (str): Path to the input HDF5 file.
        path_to_dest (str): Path to where the final file will be created
        original_shape: (tuple of int, optional): The shape of the slices the model has beeen trained on
        block_size (tuple of int, optional): The shape of the block to be used for inference. 
            Defaults to (96, 96, 96). Must be smaller than the model's slice shape.
        halo_size (tuple of int, optional): The shape of the halo to be used in each side of the block
            during inference. Defaults to (96, 96, 96). 
        raw_key (str, optional): The key in the HDF5 file that contains the raw data 
            to be used for inference. Defaults to "raw_crop".
        test_function (function, optional): function that calculates loss
        label_key (str, optional): key to reach the labels in the file; used only with test_function
    """

    # Copy the file, do checks on the extension
    if ".h5" not in path_to_file and ".hdf5" not in path_to_file:
        raise ValueError("Passed file must be HDF5 (.h5 or .hdf5), got " + path_to_file)
    if os.path.exists(path_to_dest):
        raise ValueError("Passed path to a file that already exists: ", path_to_dest)
    
    # Prepare for inference and get data
    model.eval()
    test_val = None
    try:
        copyfile(path_to_file, path_to_dest)
        with h5py.File(path_to_file, 'r') as f:
            original_crop = f[raw_key][:]
            if test_function is not None:
                label_crop = f[label_key][:]
            # If the crop is smaller than the original shape, do not use halo
            is_smaller = all(a <= b for a, b in zip(original_crop.shape, original_shape))
            if is_smaller:
                block_size = original_shape
                halo = (0,)*3

    except Exception as e:
        print(f"Failed to read original crop ({path_to_file}): ", e)
        if os.path.exists(path_to_dest):
            os.remove(path_to_dest)

    try:
        # Determine the current device: use GPU if available, else CPU
        if torch.cuda.is_available():
            gpu_ids = [torch.cuda.current_device()]
        else:
            gpu_ids = ["cpu"]
        # Predict
        item = prediction.predict_with_halo(
            original_crop,
            model,
            gpu_ids,
            block_size,
            halo,
            preprocess=minmax_norm
        )
        # TEST ONLY
        if postprocess:
            print("Postprocessing crop")
            foreground_prediction = postprocess_prediction(item[0], item[1], 0.8, True)
        else:
            foreground_prediction = item[0]

        if test_function is not None:
            test_val = test_function(original_crop, label_crop) # Somehow like this it works??
            test_val_processed = test_function(foreground_prediction, label_crop)
            print("Before refinement: ", test_val)
            print("After refinement: ", test_val_processed)
            # Should be with item[0]:
            test_val = test_function(item[0], label_crop)
            print("Check if this works: ", test_val)
        with h5py.File(path_to_dest, 'r+') as f2:
            f2.create_dataset("foreground", foreground_prediction.shape, np.float32, foreground_prediction)
            f2.create_dataset("boundary", item[0].shape, np.float32, item[1])
            # ONLY TEST
            f2.create_dataset("foreground_unprocessed", item[0].shape, np.float32, item[0])
        return test_val

    except Exception as e:
        print("Failed to test inference: ", e)
        if os.path.exists(path_to_dest):
            os.remove(path_to_dest)

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
    return (x - min)/(max - min) # !!! Write down this change

def test_inference_loss(path_to_folder, label_key = "label_crop/mito", foreground_key = "foreground", average = True, memory_saving_level = 1):
    """
    Calculate the total or average loss over all files in a specified folder using IoU and dice loss.
    Args:
        path_to_folder (str): Path to the folder containing the data files to evaluate.
        label_key (str, optional): Key to access the label data in each file. Defaults to "label_crop/mito".
        foreground_key (str, optional): Key to access the prediction data in each file. Defaults to "foreground".
        average (bool, optional): If True, returns the average loss over all files; otherwise, returns the total loss. Defaults to True.
        memory_saving_level: how many times the array is divided for the calculation; defaults to 0 (no division)
    Returns:
        float: The average or total loss computed over all files in the folder.
    """
    prediction_threshold = 0.8
    # Input checks
    if not os.path.exists(path_to_folder):
        raise ValueError("Path does not exist: ", path_to_folder)
    jaccard_metric = JaccardIndex(task="binary")

    items = directory_to_path_list(path_to_folder)
    if not items:
        raise RuntimeError(f"Folder is empty: {path_to_folder}")
    iou_score = 0
    dice_score = 0
    ious = []
    dices = []
    for item in items:
        # if os.path.basename(item) == "test_crop.h5":
        #     print("skipped large test crop")
        #     continue
        print("Calculating loss for ", os.path.basename(item))
        with h5py.File(item, "r") as f:
            if memory_saving_level > 1:
                shape = f[foreground_key].shape
                msl = memory_saving_level # Just shorter
                prediction_arr = np.array(f[foreground_key])
                label_arr = np.array(f[label_key])
                # Get steps
                step_z = shape[0] // msl
                step_y = shape[1] // msl
                step_x = shape[2] // msl
                i_score = 0
                d_score = 0
                # Iterate over dimensions
                for iz in range(msl):
                    for iy in range(msl):
                        for ix in range(msl):
                            z_start = iz * step_z
                            y_start = iy * step_y
                            x_start = ix * step_x
                            z_end = (iz + 1) * step_z if iz < msl - 1 else shape[0]
                            y_end = (iy + 1) * step_y if iy < msl - 1 else shape[1]
                            x_end = (ix + 1) * step_x if ix < msl - 1 else shape[2]
                            pred_chunk = prediction_arr[z_start:z_end, y_start:y_end, x_start:x_end]
                            label_chunk = label_arr[z_start:z_end, y_start:y_end, x_start:x_end]
                            # Binarize
                            pred_tensor = torch.from_numpy(pred_chunk).float()
                            label_tensor = torch.from_numpy(label_chunk).float()
                            pred_tensor = (pred_tensor > prediction_threshold).int()
                            label_tensor = (label_tensor > prediction_threshold).int()
                            # Flatten
                            pred_tensor_flat = pred_tensor.view(-1)
                            label_tensor_flat = label_tensor.view(-1)
                            # IoU
                            i_score += jaccard_metric(pred_tensor_flat, label_tensor_flat).item()
                            # Dice loss
                            intersection = (pred_tensor_flat * label_tensor_flat).sum().item()
                            total = pred_tensor_flat.sum().item() + label_tensor_flat.sum().item()
                            d_score += (2.0 * intersection / (total + 1e-8))
                # Normalize by number of chunks
                num_chunks = msl ** 3
                i_score /= num_chunks
                d_score /= num_chunks

            else:
                prediction_arr = np.array(f[foreground_key])
                label_arr = np.array(f[label_key])
                # Binarize 
                pred_tensor = torch.from_numpy(prediction_arr).float()
                label_tensor = torch.from_numpy(label_arr).float()
                pred_tensor = (pred_tensor > prediction_threshold).int()
                label_tensor = (label_tensor > prediction_threshold).int()
                # Flatten
                pred_tensor_flat = pred_tensor.view(-1)
                label_tensor_flat = label_tensor.view(-1)
                # IoU
                i_score = jaccard_metric(pred_tensor_flat, label_tensor_flat).item()
                # Dice loss: 1 - (2 * intersection / (sum of sizes))
                intersection = (pred_tensor_flat * label_tensor_flat).sum().item()
                total = pred_tensor_flat.sum().item() + label_tensor_flat.sum().item()
                d_score = (2.0 * intersection / (total + 1e-8))
        if average:
            iou_score += i_score
            dice_score += d_score
        else:
            ious.append(iou_score)
            dices.append(dice_score)
        print("Predicted ", os.path.basename(item), f"- IoU: {i_score}, dice: {d_score}")
    if average:
        return (iou_score / len(items), dice_score / len(items))
    return (ious, dices)

if __name__ == "__main__":
    test_folder = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_inference"
    # Test the post processing with some new files from old files
    # Calculate IOU and dice before and after!!
