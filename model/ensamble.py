import os
import numpy as np
import h5py
from skimage import morphology
from shutil import copyfile
import torch
from torch_em.util import prediction
from inference_utils import postprocess_prediction
from UNet import model, load_model


def ensemble_inference(models, shapes, sample, gpu_ids, halos, threshold_per_model = 0.4):
    """
    Args:
        models: tuple of models
        shapes: array of original train sizes of the models
        sample: the sample to verify
        halos: array of tuples with block size and halo, ex:  [((70,)*3, (29,)*3), ((88,)*3, (20,)*3)]
        gpu_ids: GPU device IDs
        threshold_per_model: threshold for each model's contribution
    """
    total_pred = np.zeros_like(sample, dtype=np.float32)
    total_bound = np.zeros_like(sample, dtype=np.float32)
    shape = sample.shape

    
    print(f"Starting ensemble inference with {len(models)} models")
    print(f"Sample shape: {shape}")
    
    for i, (model, model_size, halo) in enumerate(zip(models, shapes, halos)):
        block_size = halo[0]
        halo = halo[1]
        print(f"\nProcessing model {i+1}/{len(models)}")
        model.eval()
        print(f"Model size: {model_size}, Halo: {halo}, Block size: {block_size}")
        
        try:
            item = prediction.predict_with_halo(
                sample,
                model,
                gpu_ids,
                block_size,
                halo,
                preprocess=minmax_norm
            )
            total_pred += item[0]
            total_bound += item[1]
                    
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing model {i+1}: {e}")
            continue
    thr = threshold_per_model * len(models)


    binarized_bound = (total_bound > 0.8).astype(bool) # Changed threshold for boundaries as they are rarely wrong
    # Incorporate postprocessing
    pred = postprocess_prediction(pred, binarized_bound)

    print(f"Final threshold: {thr}")
    return pred, binarized_bound

def check_inference(models, shapes, path_to_file, path_to_dest, halos, raw_key = "raw_crop"):
    """
    Perform inference using a given model on data from an HDF5 file and save the results 
    to a new HDF5 file appending the inference results ("foreground" and "boundaries")
    to the new file.
    If the test function and label key is given, it returns the test metric.
    Args:
        models (torch.nn.Module): Tuple of PyTorch models to use for inference
        shapes: array of original train sizes of the models
        path_to_file (str): Path to the input HDF5 file.
        path_to_dest (str): Path to where the final file will be created
        halos: array of tuples with block size and halo, ex:  [((70,)*3, (29,)*3), ((88,)*3, (20,)*3)]
        raw_key (str, optional): The key in the HDF5 file that contains the raw data 
            to be used for inference. Defaults to "raw_crop".
    """

    # Copy the file, do checks on the extension
    if ".h5" not in path_to_file and ".hdf5" not in path_to_file:
        raise ValueError("Passed file must be HDF5 (.h5 or .hdf5), got " + path_to_file)
    if os.path.exists(path_to_dest):
        raise ValueError("Passed path to a file that already exists: ", path_to_dest)
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(path_to_dest), exist_ok=True)
    
    # Prepare for inference and get data
    try:
        copyfile(path_to_file, path_to_dest)
        with h5py.File(path_to_file, 'r') as f:
            original_crop = f[raw_key][:]

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
        foreground, boundary = ensemble_inference(models, shapes, original_crop, gpu_ids, halos)


        with h5py.File(path_to_dest, 'r+') as f2:
            f2.create_dataset("foreground", foreground.shape, foreground.dtype , foreground)
            f2.create_dataset("boundary", boundary.shape, boundary.dtype, boundary)
        print("Finished inference")

    except Exception as e:
        print("Failed to test inference: ", e)
        if os.path.exists(path_to_dest):
            os.remove(path_to_dest)


def minmax_norm(x):
    min = 0
    max = 255
    x = x.astype(np.float32)
    return (x - min)/(max - min) 

if __name__ == "__main__":
    models = []
    model_paths = [
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/model/checkpoints/AUNet-128-1-finetuning-DA/best.pt",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/model/checkpoints/AUNet-128-1-one-step-DA/best.pt",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/model/checkpoints/AUNet-160-1-one-step-DA/best.pt",
                   ]
    for path in model_paths:
            models.append(load_model(path, model))
    shapes = [
        (128,)*3,
        (128,)*3,
        (160,)*3,
    ]
    halos = [
        ((70,)*3, (29,)*3),
        ((88,)*3, (20,)*3),
        ((96,)*3, (32,)*3),
    ]
    check_inference(
        models,
        shapes,
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_184.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_inference/ensamble/test.h5",

    )