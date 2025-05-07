import h5py
import tifffile
import zarr
import numpy as np
import mrcfile
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


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

def read_folder_h5(f, current_path, folders_to_ignore, names):
    """
    Reads a folder and returns the folders to explore and data to keep.
    """
    # Read the data
    folders_to_save = []
    folders_to_explore = []

    # Get all items in the curent hdf5 folder
    items = list(f[current_path].keys())
    # print(f"Items found in {current_path}: {items}")
    for item in items:
        # print (f"Exploring {item} in {current_path}")
        if item in names:
            # Case we found it. Add item to dictionary as key, value
            folders_to_save.append(current_path + item + "/")
        elif item not in folders_to_ignore:
            # Pass from string to object
            item_obj = f[current_path + item]
            # Separate ccase group and dataset
            if isinstance(item_obj, h5py.Group):
                # If group, explore deeper
                folders_to_explore.append(current_path + item + "/") 
                # print(f"Found {current_path + item + '/'}" )
            # Otherwise, ignore.

    return folders_to_explore, folders_to_save

def save_folder(f, current_path):
    """
    Saves a folder to the data dictionary, or recursively open a folder until a dataset is found
    """
    data = {}
    # Read the data
    items = list(f[current_path].keys())
    for item in items:
        # print(isinstance(item, str))
        # pass from string to object in the test
        if isinstance(f[current_path + item], h5py.Dataset):
            data[current_path + item] = f[current_path + item][:]
        elif isinstance(f[current_path + item], h5py.Group):
            data = data | save_folder(f, current_path + item + "/", data)
    return data

def get_only_mito(file_path, names=["mito", "fibsem-uint8"], max_threads=None, folders_to_ignore=[]):
    """
    Takes an h5 file and returns the same file keeping only the correct mito folders in `path_to_groundtruth`, 
    renaming the file from 'x.h5' to 'x_mito.h5'.

    - Sample: "/recon-1/em/fibsem-uint8/s_"
    - Ground truth: "/recon-1/labels/groundtruth/crop___/mito/s_"
    """

    max_threads = max_threads or os.cpu_count()
    folders_to_explore = ["/"]
    folders_to_save = []
    data = {}
    f = h5py.File(file_path, "r")
    # Adapred from parallel reaed and download in utils.py
    # Multithreaded read
    print("Exploring folders in " + file_path)
    while folders_to_explore:
    # Pop up to max_threads items
        batch = []
        while folders_to_explore and (len(batch) < max_threads):
            batch.append(folders_to_explore.pop())
        print("Starting work for " + str(len(batch)) + " workers")
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(read_folder_h5, f, current_path, folders_to_ignore, names)
                for current_path in batch
            ]

            for future in as_completed(futures):
                try:
                    to_explore, to_save = future.result()
                    folders_to_explore.extend(to_explore)
                    folders_to_save.extend(to_save)
                except Exception as e:
                    print(f"Error in worker thread: {e}")

    print("Folders found: " + str(folders_to_save))
    # Save folders to dictionary
    while folders_to_save:
        batch = []
        while folders_to_save and (len(batch) < max_threads):
            batch.append(folders_to_save.pop())

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(save_folder, f, current_path)
                for current_path in batch
            ]

            for future in as_completed(futures):
                try:
                    new_data = future.result()
                    if new_data:
                        data.update(new_data)
                except Exception as e:
                    print(f"Error in worker thread: {e}")
    
    # print("Data keys: " + str(data.keys()))
    output_path = file_path.replace(".h5", "_mito.h5")
    if data:
        export_data(output_path, data)
    else:
        print("No data found")

def create_test_h5_structure(filename="test_file.h5"):
    """
    IA created function

    Creates an HDF5 file with the following structure:

    test_file.h5
    └── recon-1
        ├── em
        │   └── fibsem-uint8
        │       └── s2  → dataset
        └── labels
            └── groundtruth
                ├── crop001
                │   ├── mito
                │   │   └── s2  → dataset
                │   ├── nuc
                │   └── mem
                ├── crop002
                └── crop003
    """

    # Sample label classes and crops
    label_classes = ["mito", "nuc", "mem"]
    crop_names = ["crop001", "crop002", "crop003"]

    # Create the file
    with h5py.File(filename, "w") as f:
        # Create EM sample data
        em_path = "recon-1/em/fibsem-uint8/s2"
        f.create_dataset(em_path, data=np.random.randint(0, 255, (64, 64), dtype=np.uint8))

        # Create label data under groundtruth
        for crop in crop_names:
            for label in label_classes:
                label_path = f"recon-1/labels/groundtruth/{crop}/{label}/s2"
                f.create_dataset(label_path, data=np.random.randint(0, 2, (64, 64), dtype=np.uint8))

    print(f"Created mock HDF5 structure at: {filename}")

if __name__ == "__main__":
    names = [
        "jrc_hela-2",             # 70 GB   # 12 GB after only 8nm # 36GB in h5??
        "jrc_macrophage-2",       # 96 GB   # 15 GB     # 39GB
        "jrc_jurkat-1",           # 123 GB  # 20 GB     # 44GB
        "jrc_hela-3",             # 133 GB  # 18 GB     # 32GB
        "jrc_ctl-id8-1",          # 235 GB  # ?         # 86GB
    ]
    path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/datasets/"
    test_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/datasets/test.h5"
    # create_test_h5_structure(test_path)
    for name in names:
        get_only_mito(path + name + ".h5")
    # zarr_to_h5(path, delete=False)