import h5py
import tifffile
import zarr
import numpy as np
import mrcfile
import os
import shutil

# Most functions not made by me
# https://github.com/lufre1/synapse/blob/main/synapse/io/util.py

def read_data(path):
    """
    Read a Zarr group or array from the given path into memory.
    Returns a dictionary with dataset names as keys and NumPy arrays as values.
    """
    data = {}

    # Open the Zarr store 
    store = zarr.DirectoryStore(path)
    root = zarr.open(store, mode='r')
    # Recursively read the Zarr group
    def recursive_read(zarr_group, prefix=""):
        for name, item in zarr_group.items():
            full_name = os.path.join(prefix, name)
            if isinstance(item, zarr.Group):
                recursive_read(item, full_name)
            elif isinstance(item, zarr.Array):
                data[full_name] = item[:]

    recursive_read(root)
    return data

def export_data(export_path: str, data):
    """Export data to the specified path, determining format from the file extension.
    
    Args:
        data (np.ndarray | dict): The data to save. For HDF5/Zarr, a dict of named datasets is required.
        export_path (str): The file path where the data should be saved.
    
    Raises:
        ValueError: If the file format is unsupported or if data format does not match the expected type.
    """
    ext = export_path.lower().split(".")[-1]

    if ext == "tif":
        if isinstance(data, dict):
            data = next(iter(data.values()))
        if not isinstance(data, np.ndarray):
            raise ValueError("For .tif format, data must be a NumPy array.")
        tifffile.imwrite(export_path, data, dtype=data.dtype, compression="zlib")
        # iio.imwrite(export_path, data, compression="zlib")
    
    elif ext in {"mrc", "rec"}:
        if not isinstance(data, np.ndarray):
            raise ValueError("For .mrc and .rec formats, data must be a NumPy array.")
        with mrcfile.new(export_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(data.dtype))
    
    elif ext == "zarr":
        if not isinstance(data, dict):
            raise ValueError("For .zarr format, data must be a dictionary with dataset names as keys.")
        root = zarr.open(export_path, mode="w")
        for key, value in data.items():
            root.create_dataset(key, data=value.astype(data.dtype))

    elif ext in {"h5", "hdf5"}:
        if not isinstance(data, dict):
            raise ValueError("For .h5 and .hdf5 formats, data must be a dictionary with dataset names as keys.")
        with h5py.File(export_path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value.astype(value.dtype))
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Data successfully exported to {export_path}")

def zarr_to_h5(zarr_path: str, export_path=None, delete = True):
    """
    Convert a Zarr file to HDF5 format. 
    If export_path is not provided, the HDF5 file will be saved with the same name as the Zarr file but with a .h5 extension.
    If delete is True, the original Zarr file will be deleted after conversion.
    
    Args:
        path (str): The path to the Zarr file.
        export_path (str, optional): The path to save the HDF5 file. Defaults to None.
        delete (bool, optional): Whether to delete the original Zarr file after conversion. Defaults to True.
    """
    # Read data from Zarr
    data = read_data(zarr_path)
    
    # Export to HDF5
    if not export_path:
        # Use the same name as the Zarr file but with .h5 extension
        export_path = os.path.splitext(zarr_path)[0] + ".h5"
    export_data(export_path, data)

    if delete:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        else:
            print("Zarr file not found for deletion.") # Should not happen

def get_only_mito(file_path, path_to_groundtruth="/recon-1/labels/groundtruth/", path_to_sample="/recon-1/em/fibsem-uint8/"):
    """
    Takes an h5 file and returns the same file keeping only the correct mito folders in `path_to_groundtruth`, 
    renaming the file from 'x.h5' to 'x_mito.h5'.

    - Sample: "/recon-1/em/fibsem-uint8/s_"
    - Ground truth: "/recon-1/labels/groundtruth/crop___/mito/s_"
    """

    data = {}
    # Get the list of the paths to follow
    parts = list(filter(None, path_to_groundtruth.split("/")))
    parts2 = list(filter(None, path_to_sample.split("/")))
    print(parts)
    print(parts2)
    # ['recon-1', 'labels', 'groundtruth']
    # ['recon-1', 'em', 'fibsem-uint8']
    
    # Case 1: find sample
    
    
    # Open file
    with h5py.File(file_path, "r") as f:
        # Function to iterate in the h5
        def collect_items(h5_group, path=""):
            # Check all items
            for key in h5_group:
                item = h5_group[key]
                full_path = os.path.join(path, key)
                # If we find a group, open it
                if isinstance(item, h5py.Group):
                    # If we are still looking, proceed
                    if parts.__len__ > 0:
                        if key == parts.pop():
                            collect_items(item, full_path)
                    # If we got in the place after the path, open all crops
                    else:
                        # In this case, we should have a list of crops. Check if there is mito in all of them and save that.
                        for crops in item:
                            if "mito" in crops:
                                # If it is, save it
                                data[full_path] = item[crops][()]
                # Fully discard everything if it is not the group we want AND we are still looking

    output_path = file_path.replace(".h5", "_mito.h5")
    export_data(output_path, data)
    print(f"Finished transforming h5 {output_path}")

if __name__ == "__main__":
    print("[zarr_utils.py]")
    path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/datasets/jrc_hela-2.h5"
    get_only_mito(path)
    # zarr_to_h5(path, delete=False)