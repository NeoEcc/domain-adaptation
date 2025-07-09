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
    except Exception as e:
        print("Failed to write file ", e)

# # readme.md information        
# Similarly, `data.json` is the product of `read-zattrs`, which takes a list of `.zattrs` files downloaded by `get-zattrs` and collects the important metadata of the zarrs available for download.  

# ```
# {
#     name: foo,
#     voxel_size: [z, y, x],
#     has_groundtruth: True,
#     has_inference: False
# }
# ```
# It also stores the number of datasets with inference, ground truth and both:
# ```
# {
#     "inference": x, 
#     "groundtruth": y, 
#     "groundtruth_and_inference": z,
# }
# ```

# crop_from_dataset.py
    # Verifying which samples have mitochondria
    
good_samples = []
for file in os.listdir("/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/labeled_crops/"):
    path_to_file = f"/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/labeled_crops/{file}"
    with h5py.File(path_to_file, 'r') as f:
        try:
            print(os.path.basename(path_to_file))
            print("Rand element of labels: " + str(f["label_crop/mito"][40][40][40]))
            if np.any(f["label_crop/mito"]):
                print("Labels ok")
                good_samples.append(file)
            print()
        except Exception as e:
            print(os.path.basename(path_to_file), ": failed to open dataset mito: ", e)
print()
print("Good samples with mito: ")
good_samples.sort()
print(good_samples)

# Extract test crops with inference
#!/usr/bin/env python3
"""
Script to extract overlapping raw and label crops from datasets with different coordinate systems.

Raw dataset: (8932, 12728, 12747) (zyx)
Label dataset: (5000, 5000, 5000) (zyx)
Label coordinates in raw space: 30984, 30912, 15728 - 70984, 70912, 55728 (xyz, nm, 8nm/pixel)
"""

import numpy as np
import h5py
import zarr
import s3fs
import z5py
from typing import Tuple, Optional

def h5_from_bucket(zarr_path, zarr_key, hdf5_path, roi):
    """
    Given the details in a bucket, downloads only the specified part of a dataset (roi) and
    prints the dataset in an hdf5 file.
    """
    fs = s3fs.S3FileSystem(anon=True)
    try:
        # For Zarr files
        store = zarr.storage.FSStore(zarr_path, fs=fs)
        dataset = zarr.open(store, mode='r', path=zarr_key)
        print(f"Successfully opened Zarr dataset: {dataset.shape}, dtype: {dataset.dtype}")
        
        # Extract roi
        roi_data = dataset[roi]
        print(f"ROI shape: {roi_data.shape}")
        
    except Exception as e:
        print(f"Zarr processing failed: {e}")
        raise
    
    # Save if path provided
    if hdf5_path is not None:
        import os
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        with h5py.File(hdf5_path, 'w') as h5_file:
            h5_file.create_dataset("data", data=roi_data, compression="gzip")
            h5_file.attrs["source_path"] = zarr_path
            h5_file.attrs["source_key"] = zarr_key
            h5_file.attrs["roi"] = str(roi)
        print(f"Saved to {hdf5_path}")
    
    return roi_data

def n5_to_hdf5(n5_path, hdf5_path, roi=None):
    """Convert N5 file to HDF5 with optional roi"""
    print(f"Converting N5 to HDF5")
    try:
        with z5py.File(n5_path, 'r') as n5_file:
            ds_names = list(n5_file.keys())
            if not ds_names:
                raise ValueError("No datasets found in N5 file.")
            ds_name = ds_names[0]
            n5_ds = n5_file[ds_name]
            # ROI
            if roi is not None:
                data = n5_ds[roi]
                print("Shape: ", n5_ds.shape, " - after ROI: ", data.shape)
            else:
                data = n5_ds[:]
                print("Shape: ", data.shape)
            
            if hdf5_path is not None:
                import os
                os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
                with h5py.File(hdf5_path, 'w') as h5_file:
                    h5_file.create_dataset(ds_name, data=data, compression="gzip")
                
            print(f"Successfully converted to HDF5: {hdf5_path}")
            return data
    except Exception as e:
        try:
            print(f"Conversion failed first time: {e}")
            n5_store = zarr.N5Store(n5_path)
            group = zarr.open(n5_store, mode='r')
            # Recursively find the first array
            if isinstance(group, zarr.Array):
                print("Group shape: ", group.shape)
                if roi is not None:
                    data = group[roi]
                    print("Reduced shape: ", data.shape)
                else:
                    data = group[:]
                if hdf5_path is not None:
                    import os
                    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
                    with h5py.File(hdf5_path, 'w') as h5_file:
                        h5_file.create_dataset("data", data=data, compression="gzip")
                
            print(f"Successfully converted to HDF5: {hdf5_path}")
            return data
        except Exception as e2:
            print("Failed to convert twice: ", e, e2)
            raise e2

def nm_to_voxel(nm_coords: Tuple[int, int, int], voxel_size: int = 8) -> Tuple[int, int, int]:
    """Convert nanometer coordinates to voxel coordinates"""
    return tuple(coord // voxel_size for coord in nm_coords)

def xyz_to_zyx(xyz_coords: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert XYZ coordinates to ZYX coordinates"""
    return (xyz_coords[2], xyz_coords[1], xyz_coords[0])

def create_overlapping_roi(raw_shape: Tuple[int, int, int], 
                          label_shape: Tuple[int, int, int],
                          label_start_xyz_nm: Tuple[int, int, int],
                          label_end_xyz_nm: Tuple[int, int, int],
                          crop_size: Tuple[int, int, int],
                          voxel_size: int = 8) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Create ROIs for both raw and label datasets that overlap in the same physical space.
    
    Args:
        raw_shape: Shape of raw dataset (z, y, x)
        label_shape: Shape of label dataset (z, y, x)  
        label_start_xyz_nm: Start coordinates of label in raw space (x, y, z) in nm
        label_end_xyz_nm: End coordinates of label in raw space (x, y, z) in nm
        crop_size: Desired crop size (z, y, x)
        voxel_size: Voxel size in nm
        
    Returns:
        Tuple of (raw_roi, label_roi)
    """
    
    # Convert nm coordinates to voxel coordinates
    label_start_voxel = nm_to_voxel(label_start_xyz_nm, voxel_size)
    label_end_voxel = nm_to_voxel(label_end_xyz_nm, voxel_size)
    
    # Convert XYZ to ZYX for array indexing
    label_start_zyx = xyz_to_zyx(label_start_voxel)
    label_end_zyx = xyz_to_zyx(label_end_voxel)
    
    print(f"Raw dataset shape (ZYX): {raw_shape}")
    print(f"Label dataset shape (ZYX): {label_shape}")
    print(f"Label region in raw space (ZYX voxels): {label_start_zyx} to {label_end_zyx}")
    
    # Calculate the overlap region dimensions
    overlap_size = tuple(end - start for start, end in zip(label_start_zyx, label_end_zyx))
    print(f"Label region size (ZYX): {overlap_size}")
    
    # Ensure crop size fits within both datasets
    max_crop_size = []
    for i, (raw_dim, label_dim, overlap_dim) in enumerate(zip(raw_shape, label_shape, overlap_size)):
        max_size = min(raw_dim, label_dim, overlap_dim, crop_size[i])
        max_crop_size.append(max_size)
    
    final_crop_size = tuple(max_crop_size)
    print(f"Final crop size (ZYX): {final_crop_size}")
    
    # Choose a random position within the overlap region for the crop
    import random
    crop_start_raw = []
    crop_start_label = []
    
    for i in range(3):
        # Maximum starting position in raw coordinates
        max_start_raw = label_end_zyx[i] - final_crop_size[i]
        min_start_raw = label_start_zyx[i]
        
        if max_start_raw < min_start_raw:
            raise ValueError(f"Crop size {final_crop_size[i]} is too large for dimension {i}")
        
        # Random start position in raw coordinates
        start_raw = random.randint(min_start_raw, max_start_raw)
        crop_start_raw.append(start_raw)
        
        # Corresponding start position in label coordinates
        start_label = start_raw - label_start_zyx[i]
        crop_start_label.append(start_label)
    
    # Create ROIs
    raw_roi = tuple(slice(start, start + size) for start, size in zip(crop_start_raw, final_crop_size))
    label_roi = tuple(slice(start, start + size) for start, size in zip(crop_start_label, final_crop_size))
    
    print(f"Raw ROI (ZYX): {raw_roi}")
    print(f"Label ROI (ZYX): {label_roi}")
    
    return raw_roi, label_roi

def extract_overlapping_crops(raw_shape: Tuple[int, int, int] = (8932, 12728, 12747),
                             label_shape: Tuple[int, int, int] = (5000, 5000, 5000),
                             label_start_xyz_nm: Tuple[int, int, int] = (30984, 30912, 15728),
                             label_end_xyz_nm: Tuple[int, int, int] = (70984, 70912, 55728),
                             crop_size: Tuple[int, int, int] = (1024,)*3,
                             output_dir: str = "./files/overlapping_crops/",
                             crop_name: str = "overlap_crop"):
    """
    Extract overlapping crops from raw and label datasets.
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROIs
    raw_roi, label_roi = create_overlapping_roi(
        raw_shape, label_shape, label_start_xyz_nm, label_end_xyz_nm, crop_size
    )
    
    # Extract raw crop
    print("\nExtracting raw crop...")
    raw_output_path = os.path.join(output_dir, f"{crop_name}_raw.h5")
    raw_data = h5_from_bucket(
        "janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr", 
        "recon-1/em/fibsem-uint8/s0", 
        raw_output_path, 
        raw_roi
    )
    
    # Extract label crop
    print("\nExtracting label crop...")
    label_output_path = os.path.join(output_dir, f"{crop_name}_labels.h5")
    label_data = n5_to_hdf5(
        "./files/mito_bag_seg.n5/", 
        label_output_path, 
        label_roi
    )
    
    # Create combined file
    print("\nCreating combined file...")
    combined_output_path = os.path.join(output_dir, f"{crop_name}_combined.h5")
    with h5py.File(combined_output_path, 'w') as h5_file:
        h5_file.create_dataset("raw", data=raw_data, compression="gzip")
        h5_file.create_dataset("labels", data=label_data, compression="gzip")
        h5_file.attrs["voxel_size"] = (8, 8, 8)
        h5_file.attrs["raw_roi"] = str(raw_roi)
        h5_file.attrs["label_roi"] = str(label_roi)
        h5_file.attrs["label_region_xyz_nm"] = f"{label_start_xyz_nm} to {label_end_xyz_nm}"
        h5_file.attrs["source_raw"] = "janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr"
        h5_file.attrs["source_labels"] = "./files/mito_bag_seg.n5/"
    
    print(f"Combined file saved to: {combined_output_path}")
    print(f"Raw crop shape: {raw_data.shape}")
    print(f"Label crop shape: {label_data.shape}")
    
    return raw_data, label_data, raw_roi, label_roi

def verify_overlap(raw_roi, label_roi, label_start_xyz_nm, voxel_size=8):
    """Verify that the crops actually overlap in physical space"""
    label_start_zyx = xyz_to_zyx(nm_to_voxel(label_start_xyz_nm, voxel_size))
    
    # Convert label ROI to raw coordinate system
    label_roi_in_raw = tuple(
        slice(label_slice.start + label_start_zyx[i], 
              label_slice.stop + label_start_zyx[i])
        for i, label_slice in enumerate(label_roi)
    )
    
    print(f"\nVerification:")
    print(f"Raw ROI: {raw_roi}")
    print(f"Label ROI in raw coordinates: {label_roi_in_raw}")
    
    # Check if they're identical (they should be for overlapping crops)
    overlap_verified = all(
        raw_slice.start == label_slice.start and raw_slice.stop == label_slice.stop
        for raw_slice, label_slice in zip(raw_roi, label_roi_in_raw)
    )
    
    print(f"Overlap verified: {overlap_verified}")
    return overlap_verified

if __name__ == "__main__":
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    # Extract overlapping crops
    print("Extracting overlapping crops...")
    raw_data, label_data, raw_roi, label_roi = extract_overlapping_crops()
    
    # Verify overlap
    verify_overlap(raw_roi, label_roi, (30984, 30912, 15728))
    
    print("Finished extracting crops")



# Mitochondria download
from utils import get_filtered_from_bucket, get_folder_parallel

# Hardcoded list of all datasets available with mitochondria
names = [
    "jrc_hela-h89-1",
    "jrc_hela-h89-2",
    "jrc_mus-kidney",
    "jrc_mus-pancreas-1",
    "jrc_mus-pancreas-2",
    "jrc_mus-pancreas-3",
    "jrc_hela-21",            # 9.8 GB
    "jrc_fly-larva-1",        # 9.72 TB
    "jrc_hela-22",            # 16 GB
    "jrc_ctl-id8-5",          # 18.3 GB
    "jrc_ctl-id8-2",          # 28.2 GB
    "jrc_hela-1",             # 31 GB
    "jrc_ctl-id8-4",          # 32.7 GB
    "jrc_ctl-id8-3",          # 39.2 GB
    "jrc_hela-bfa",           # 14 GB
    "jrc_hela-2",             # 70 GB
    "jrc_macrophage-2",       # 96 GB
    "jrc_cos7-11",            # 90.6 GB
    "jrc_jurkat-1",           # 123 GB
    "jrc_hela-3",             # 133 GB
    "jrc_ctl-id8-1",          # 235 GB
    "jrc_mus-sc-zp105a",      # 258 GB
    "jrc_mus-sc-zp104a",      # 340 GB
    "jrc_hela-4",             # 318 GB
    "jrc_dauer-larva",        # 315 GB
    "jrc_ccl81-covid-1",      # 356 GB
    "jrc_choroid-plexus-2",   # 507 GB
    "jrc_mus-liver",          # 1.12 TB
    "jrc_fly-acc-calyx-1",    # 4.81 TB
    "jrc_fly-fsb-1",          # 2.45 TB
    "jrc_fly-mb-z0419-20",    # 2.45 TB
    "jrc_sum159-1",           # 13.9 TB
]

names_with_labels = [
    "jrc_hela-2",             # 70 GB  
    "jrc_macrophage-2",       # 96 GB   
    "jrc_jurkat-1",           # 123 GB  
    "jrc_hela-3",             # 133 GB  
    "jrc_ctl-id8-1",          # 235 GB  
    "jrc_mus-kidney"          # unknown 
    "jrc_mus-liver",          # 1.12 TB 
    "jrc_sum159-1",           # 13.9 TB
]

folders_to_ignore = [
    "cell_seg",
    "cent_seg",
    "er_seg",
    "golgi_seg",
    "lyso_seg",
    "nucleus_seg",
]
b = q3.Bucket("s3://janelia-cosem-datasets")
get_folder_parallel(
    b, 
    "jrc_ctl-id8-2/jrc_ctl-id8-2.zarr/", 
    "/user/niccolo.eccel/u15001/example_dataset/jrc_ctl-id8-2.zarr/", 
    "s0", 
    16, 
    folders_to_ignore, 
)