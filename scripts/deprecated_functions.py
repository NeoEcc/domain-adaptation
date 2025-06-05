import h5py
import numpy as np
import os
import torch_em
import shutil
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
import quilt3 as q3

#
# Functions that are not used anymore but could still be useful in the future
#

# model_utils.py

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

# utils.py


def check_dataset_empty(path_to_file, subpath = "label_crop/mito", copy_path = None, remove = False):
    """
    Verifies whether an HDF5 file contains nonempty dataset in subpath.
    If it is empty, verifies whether the "label_crop/all" dataset contains 
    mito, i.e. id 3.

    If copy_path is not none, copy all valid crops there.

    If remove is True, delete original files from folder.

    Args:
        path_to_file (str): Path to the HDF5 file to check.
        subpath (str, optional): HDF5 dataset path to check for non-emptiness. Defaults to "label_crop/mito".
        copy_path (str, optional): If provided, valid files are copied to this path. Defaults to None.
        remove (bool, optional): If True, files without valid mitochondria labels are deleted. Defaults to False.    
    """
    ids = [3, 4] # IDs that refer to mitochondria
    file_name = os.path.basename(path_to_file)
    
    with h5py.File(path_to_file, 'r') as f:
        # Case there is an all crop
        if "label_crop/all" in f.keys():
            list_of_ids = np.unique(f["label_crop/all"])
            print(list_of_ids)
            if subpath not in f:
                if ids in list_of_ids:
                    # Case we have mito in all but not in mito, worst case
                    print(file_name + " lacks mito label but has some in \"all\"")
                else:
                    print(file_name + " has no reference to mito")
                    if remove:
                        os.remove(path_to_file)
            # Case the label is all zeros
            elif not np.any(f[subpath]):
                print("Empty mito label")
                # print(np.unique(f["label_crop/mito"], return_counts=True))
                if remove:
                    os.remove(path_to_file)
                
                
                return False
            if copy_path is not None:
                print("Copying file: ", path_to_file)
                shutil.copyfile(path_to_file, copy_path)
            # Else return true
            return True

# zarr_utils.py

def download_zattr(name, download_path, bucket):
    """
    Only for parallelization in get_zattrs
    Downloads the .zattrs file for a given name from the bucket to the specified download path.
    """
    try:
        b = q3.Bucket(bucket)
        b.fetch(f"{name}/{name}.zarr/recon-1/em/fibsem-uint8/.zattrs", f"{download_path}/{name}.zattrs")
        print(f"Downloaded {name}.zattrs")
    except Exception as e:
        print(f"Failed to download {name}.zattrs: {e}")

def get_zattrs(names, download_path, bucket, max_threads=None):
    """
    Downloads all .zattrs files for a list of names, in parallel.
    """
    max_threads = max_threads or min(32, len(names)) or 4  # Reasonable default

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(download_zattr, name, download_path, bucket)
            for name in names
        ]
        for future in as_completed(futures):
            future.result()  # Triggers any exceptions

def read_zattrs(folder_path, bucket_str):
    """
    Reads the content of all files in the path and prints the voxel size and name in a json file as 
    ```
    name: foo,
    voxel_size: [z, y, x],
    has_groundtruth: True,
    has_inference: False
    ```
    Takes as input the path to the folder where the files are stored and the bucket string.
    """
    all_data = []
    groundtruth = 0
    inference = 0
    groundtruth_and_inference = 0
    for file in os.listdir(folder_path):
        print("Attempting read of " + file )
        if os.path.splitext(file)[1] != ".zattrs":
            continue
        # Get name and voxel size
        b = q3.Bucket(bucket_str)
        with open(f"{folder_path}{file}") as f:
            data = json.load(f)
        name = os.path.splitext(file)[0]

        voxel_size = data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        # Check if there is a labels folder, and if there is a ground truth folder or inference folder
        has_groundtruth = False
        has_inference = False
        try:
            data = b.ls(f"{name}/{name}.zarr/recon-1/labels/")
            prefixes, keys, _ = data  # Unpack the tuple
            
            # Check in 'Prefix' fields
            # Probably not needed?
            for prefix_entry in prefixes:
                if "groundtruth" in prefix_entry.get('Prefix', ''):
                    has_groundtruth = True
                if "inference" in prefix_entry.get('Prefix', ''):
                    has_inference = True
            
            # Check in 'Key' fields
            for key_entry in keys:
                if "groundtruth" in prefix_entry.get('Key', ''):
                    has_groundtruth = True
                if "inference" in prefix_entry.get('Key', ''):
                    has_inference = True
        except Exception as e:
            print("No labels folder found for " + name + ": " + str(e))
        if has_groundtruth:
            groundtruth += 1
            if has_inference:
                groundtruth_and_inference += 1
        if has_inference:
            inference += 1
        # Make json
        json_data = {
            'name': file,
            'voxel_size': voxel_size,
            'has_groundtruth': has_groundtruth,
            'has_inference': has_inference,
        }
        print(json_data)
        all_data.append(json_data)
        
    overall_json = {
        'inference': inference,
        'groundtruth': groundtruth,
        'groundtruth_and_inference': groundtruth_and_inference
    }
    all_data.append(overall_json)
    try:
        with open('../files/zattrs/data.json', 'w') as outfile:
            json.dump(overall_json, outfile)
            json.dump(all_data, outfile)
    except:
        print("Failed to write file")