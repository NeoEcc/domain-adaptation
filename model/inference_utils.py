import h5py
import numpy as np
import os
import time
import torch

from model_utils import directory_to_path_list, minmax_norm
from torch_em.util import prediction
from torchmetrics import JaccardIndex 
from shutil import copyfile
from skimage import morphology
from scipy.ndimage import label
import napari_segment_blobs_and_things_with_membranes as nsbatwm

import elf.parallel as parallel

from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes

# synapse-net's version
def _run_segmentation(
    foreground, boundaries, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    area_threshold=5000,
):
    t0 = time.time()
    dist = parallel.distance_transform(
        boundaries < boundary_threshold, halo=halo, verbose=verbose, block_shape=block_shape
    )
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    # Get the segmentation via seeded watershed.
    t0 = time.time()
    seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    t0 = time.time()
    hmap = (dist.max() - dist) / dist.max()
    hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()
    mask = (foreground + boundaries) > 0.5

    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    seg = apply_size_filter(seg, min_size, verbose=verbose, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold)
    return seg

def _postprocess_seg_3d(seg, area_threshold=1000, iterations=4, iterations_3d=8):
    # Structure lement for 2d dilation in 3d.
    structure_element = np.ones((3, 3))  # 3x3 structure for XY plane
    structure_3d = np.zeros((1, 3, 3))  # Only applied in the XY plane
    structure_3d[0] = structure_element

    props = regionprops(seg)
    for prop in props:
        # Get bounding box and mask.
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:3], prop.bbox[3:]))
        mask = seg[bb] == prop.label

        # Fill small holes and apply closing.
        mask = remove_small_holes(mask, area_threshold=area_threshold)
        mask = np.logical_or(binary_closing(mask, iterations=iterations), mask)
        mask = np.logical_or(binary_closing(mask, iterations=iterations_3d, structure=structure_3d), mask)
        seg[bb][mask] = prop.label

    return seg

def apply_size_filter(
    segmentation: np.ndarray,
    min_size: int,
    verbose: bool = False,
    block_shape = (128, 256, 256),
) -> np.ndarray:
    """Apply size filter to the segmentation to remove small objects.

    Args:
        segmentation: The segmentation.
        min_size: The minimal object size in pixels.
        verbose: Whether to print runtimes.
        block_shape: Block shape for parallelizing the operations.

    Returns:
        The size filtered segmentation.
    """
    if min_size == 0:
        return segmentation
    t0 = time.time()
    if segmentation.ndim == 2 and len(block_shape) == 3:
        block_shape_ = block_shape[1:]
    else:
        block_shape_ = block_shape
    ids, sizes = parallel.unique(segmentation, return_counts=True, block_shape=block_shape_, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return segmentation

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
    pred = _run_segmentation(foreground, boundaries, False, 500, (128,)*3)
    return pred

    boundaries = None
    # Binarize the prediction
    prediction = (foreground > threshold)
    prediction = morphology.remove_small_holes(prediction, 5000) # 5000 in synapse, was 512
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
    
def test_inference_loss(path_to_folder, label_key = "label_crop/mito", foreground_key = "foreground", average = True, memory_saving_level = 1):
    """
    Calculate the total or average loss over all files in a specified folder using IoU and dice loss.
    Args:
        path_to_folder (str): Path to the folder containing the data files to evaluate.
        label_key (str, optional): Key to access the label data in each file. Defaults to "label_crop/mito".
        foreground_key (str, optional): Key to access the prediction data in each file. Defaults to "foreground".
        average (bool, optional): If True, returns the average loss over all files; otherwise, returns the total loss. Defaults to True.
        memory_saving_level: how many times the array is divided for the calculation; defaults to 1 (no division). 
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
        
        with h5py.File(item, "r") as f:
            if memory_saving_level <= 1:
                print("Calculating loss for ", os.path.basename(item))
                prediction_arr = np.array(f[foreground_key])
                # Load and binarize 
                pred_tensor = torch.from_numpy(prediction_arr).float()
                pred_tensor = (pred_tensor > prediction_threshold).bool()
                
                label_arr = np.array(f[label_key])
                label_tensor = torch.from_numpy(label_arr).float()
                label_tensor = (label_tensor > prediction_threshold).bool()
                # Flatten
                pred_tensor_flat = pred_tensor.view(-1)
                label_tensor_flat = label_tensor.view(-1)
                # IoU
                i_score = jaccard_metric(pred_tensor_flat, label_tensor_flat).item()
                # Dice loss: 1 - (2 * intersection / (sum of sizes))
                intersection = (pred_tensor_flat * label_tensor_flat).sum().item()
                total = pred_tensor_flat.sum().item() + label_tensor_flat.sum().item()
                d_score = (2.0 * intersection / (total + 1e-8))
            else:
                print("Calculating loss in chunks for ", os.path.basename(item))
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

def check_inference(model, path_to_file, path_to_dest, original_shape = (128,)*3, block_size = (96,)*3, halo = (16,)*3, 
                    raw_key = "raw_crop", postprocess = False):
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
        label_key (str, optional): key to reach the labels in the file; used only with test_function
    """

    # Copy the file, do checks on the extension
    if ".h5" not in path_to_file and ".hdf5" not in path_to_file:
        raise ValueError("Passed file must be HDF5 (.h5 or .hdf5), got " + path_to_file)
    if os.path.exists(path_to_dest):
        raise ValueError("Passed path to a file that already exists: ", path_to_dest)
    
    # Prepare for inference and get data
    model.eval()
    try:
        copyfile(path_to_file, path_to_dest)
        with h5py.File(path_to_file, 'r') as f:
            original_crop = f[raw_key][:]
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
            if max(item[0].shape) > 10240: # Already fails earlier though - this might be useless asking for 32-64gb
                print("Large size crop - dividing it in 8")
                # Divide the crop into 8 sub-blocks
                z_mid = item[0].shape[0] // 2
                y_mid = item[0].shape[1] // 2
                x_mid = item[0].shape[2] // 2
                foreground_prediction = np.zeros_like(item[0])
                for iz, (z0, z1) in enumerate([(0, z_mid), (z_mid, item[0].shape[0])]):
                    for iy, (y0, y1) in enumerate([(0, y_mid), (y_mid, item[0].shape[1])]):
                        for ix, (x0, x1) in enumerate([(0, x_mid), (x_mid, item[0].shape[2])]):
                            fg_block = item[0][z0:z1, y0:y1, x0:x1]
                            bd_block = item[1][z0:z1, y0:y1, x0:x1]
                            processed_block = postprocess_prediction(fg_block, bd_block, 0.8, True)
                            foreground_prediction[z0:z1, y0:y1, x0:x1] = processed_block
            else:
                foreground_prediction = postprocess_prediction(item[0], item[1], 0.8, True)
        else:
            foreground_prediction = item[0]

        with h5py.File(path_to_dest, 'r+') as f2:
            f2.create_dataset("foreground", foreground_prediction.shape, np.float32, foreground_prediction)
            f2.create_dataset("boundary", item[0].shape, np.float32, item[1])
            # ONLY TEST
            f2.create_dataset("foreground_unprocessed", item[0].shape, np.float32, item[0])

    except Exception as e:
        print("Failed to test inference: ", e)
        if os.path.exists(path_to_dest):
            os.remove(path_to_dest)
