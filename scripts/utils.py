import time
import os
import quilt3 as q3
import json
import h5py
import torch
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from zarr_utils import zarr_to_h5, export_data
from skimage.transform import resize, rescale
from scipy.ndimage import label, median_filter


# Tried from https://hdmf-zarr.readthedocs.io/en/dev/tutorials/plot_convert_nwb_hdf5.html

def get_best_size(voxel_size, target_size, tollerance = 0.4) -> int:
    """
    Returns an int between 0 and 8 to get the correct downsampling to be close enough to target size, which should be a power of 2. 

    Close enough is defined by tollerance bercentage, which should be between 0.01 and 0.5. 
    Supose we have a voxel list of (3.9,3.9,3.9), target_size = 8, tollerance percentage = 0.0.
    The algorithm will return a compression of 2, equivalent to a voxel size of (16,16,16).
    If there is some tollerance, e.g. 0.25, it will return a compression of 1, equivalent to a voxel size of (7.8,7.8,7.8).
    """
    compression = 0
    if voxel_size < 0.01:
        raise ValueError("Voxel size should be greater than 0.01") # To prevent loops
    while voxel_size < (target_size * (1.0 - tollerance)) and compression < 8:
        compression += 1
        voxel_size *= 2.0

    return compression

def download_folder(bucket, folder_path, download_path):
    """
    Downloads the folder from the folder_path in the bucket to the download_path.
    Required for parallelization, not to be used as-is.
    """
    try:    # Differentiate between folders and files
        name = os.path.basename(folder_path)
        if name == "":
            download_path = download_path + "/"
        bucket.fetch(folder_path, f"{download_path}")
    except Exception as e:
        print(f"Failed to download {folder_path}: {e}")

def read_folder(bucket, path, download_path, folders_to_ignore, name):
    """
    Scans the path for folders, looks for the name:
    - if the name is found, add it to folders_to_download; 
    - otherwise, add all folders to folders_to_explore except those in folders_to_ignore
    return the tuple with the two lists.
    
    Required for parallelization, not to be used as-is.
    """
    folders_to_explore = []
    folders_to_download = []
    zarr_root_keys = [".zgroup", ".zattrs"]

    
    try:
        # Get the list of folders in the path
        folders = [entry['Prefix'] for entry in bucket.ls(path)[0]]
        files = [entry['Key'] for entry in bucket.ls(path)[1]]
        # If empty, end thread
        if not folders and not files:
            return ([],[])
        # Skip if empty
        # print("Exploring " + path)
        # Get target folder path
        # Get metadata files .zgroup and .zattrs
        for key in zarr_root_keys:
            if f"{path}{key}" in files:
                folders_to_download.append((f"{path}{key}", f"{download_path}/{key}"))
                # print(f"New file: {download_path}/{key}")
        target = f"{path}{name}/"
        # If the target is there, add it to the list; 
        # Proceed  only otherwise, not to explore other s_ folders
        if target in folders:
            # print("Found " + name + f" folder at {path}.")
            folders_to_download.append((target, os.path.join(download_path, name)))
        else:
            for folder in folders:
                if os.path.basename(os.path.normpath(folder)) in folders_to_ignore:
                    # print("Skipping " + folder)
                    continue
                nested_download_path = os.path.join(download_path, os.path.basename(os.path.normpath(folder)))
                folders_to_explore.append((folder, nested_download_path))
    except Exception as e:
        print(f"Error exploring {path}: {e}")
    return (folders_to_explore, folders_to_download)

def get_folder_parallel(bucket, path, download_path, name="s0", max_threads=None, folders_to_ignore=[], file_to_read=None, write_path=None):
    """
    Explores folders in buckets and downloads the required folders using as many threads as given.
    Ignores folders with names in the list folders_to_ignore and adds all levels aside from the 
    specified one, if it is of the kind s0-s8.
    
    If the path to a file is provided, the exploration step will be skipped in favor of using the 
    content of the file. 

    Args:
        bucket: bucket object where to download from
        path: path inside the bucket (END WITH "/")
        download_path: path where to download the file, of the type base/files/name.zarr 
        name: name of the folder to find
        max_threads: number of threads to use for downloading
        folders_to_ignore: list of folders to ignore
        file_to_read: (optional) path to a file with the folders to download, in the format `folder_path \t download_path`
        write_path: (optional) path to print file. Will not create file if it is "None". 
    """
    max_threads = max_threads or os.cpu_count()
    folders_to_explore = [(path, download_path)]
    folders_to_download = []
    start = time.time()
    # The check later will fail if the dataset does not have the required level of downsampling:
    # This will make the search eternally long going through all the folders in s0 and so on
    # An explicit check is required
    other_folders = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    if name in other_folders:
        other_folders.remove(name)
        folders_to_ignore.extend(other_folders)
    # If there is a file to read, read it and skip the exploration
    # Otherwise, explore the folders
    if not file_to_read:
        # Breadth-first traversal to collect folders
        print("Exploring folders...")
        while folders_to_explore:
        # Pop up to max_threads items
            batch = []
            while folders_to_explore and (len(batch) < max_threads):
                batch.append(folders_to_explore.pop())

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Each thread gets a different path from the batch
                futures = [
                    executor.submit(read_folder, bucket, current_path, current_download_path, folders_to_ignore, name)
                    for current_path, current_download_path in batch
                ]

                for future in as_completed(futures):
                    try:
                        folders_explore, folders_download = future.result()
                        folders_to_explore.extend(folders_explore)
                        folders_to_download.extend(folders_download)
                    except Exception as e:
                        print(f"Error in worker thread: {e}")

        mid = time.time()

        print("Finished exploring folders. Found "+ str(len(folders_to_download)) + f" folders to download in {mid-start}s. Downloading...")
        # Save list to file
        if write_path is not None: 
            # Generate path
            # file_path = f"{os.path.dirname(os.path.dirname(download_path))}/files/txt/to_download_{os.path.splitext(os.path.basename(download_path[:-1]))[0]}.txt"
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w") as f: 
                for folder, path in folders_to_download:
                    f.write(f"{folder}\t{path}\n")
        
    else:
        print("Reading file " + file_to_read)
        try:
            with open(file_to_read, "r") as f:
                for line in f:
                    folder, path = line.strip().split("\t")
                    folders_to_download.append((folder, path))
            print(f"Found {len(folders_to_download)} folders to download from file.")
        except Exception as e:
            print(f"Error reading file {file_to_read}: {e}")
            return
    # Download the folders in parallel
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(download_folder, bucket, folder, path) for folder, path in folders_to_download]
        for future in as_completed(futures):
            future.result()  # trigger exceptions if any
    end = time.time()
    print("Finished downloading " + str(len(folders_to_download)) + f" folders in {end - start}s (Found in {mid - start}s).")

def get_filtered_from_bucket(names, path, bucket_str, inference = False, target_size = 8, max_threads = None):
    """
    Downloads files from a bucket from the `names` list following these steps:
    - Iterate over the list of names
    - Check whether there exists a groundtruth folder (or an inference folder), otherwise skip to next name;
    - Download and read the .zargs file, get the voxel size
    - Decide the downscale level to get closer to the target size
    - Download the correct downscaled version of the dataset
    - Convert to h5

    Args:
        names (list): List of names to download
        path (str): Path to the folder where the data will be downloaded. ENDS WITH "/"
        bucket_str (str): Bucket name
        inference (bool): Whether to download datasets that have no groundtruth but have inference
        target_size (int): Target size in nm between voxels
    """
    b = q3.Bucket(bucket_str)
    # Iterate
    for name in names: 
        print("[get_some]: Handling " + name)
        try:
            # Check if there is a groundtruth 
            # If not, check for inference if required
            # Otherwise, continue to next file
            try:
                data = b.ls(f"{name}/{name}.zarr/recon-1/labels/groundtruth/")
            except:
                print("groundtruth not found for " + name)
                if inference: 
                    try:
                        data = b.ls(f"{name}/{name}.zarr/recon-1/labels/inference/")
                    except:
                        print("inference not found for " + name)
                        continue
                else:  
                    continue
            # If here, dataset was deemed downloadable. 
            # Get the attributes file: read the voxel size and decide the scale level. 
            if not os.path.isfile(f"{path}/zattrs/{name}.zattrs"):
                    b.fetch(f"{name}/{name}.zarr/recon-1/em/fibsem-uint8/.zattrs", f"{path}/zattrs/{name}.zattrs")
            try:
                with open(f"{path}/zattrs/{name}.zattrs") as f:
                    data = json.load(f)
                name = os.path.splitext(name + ".zarr")[0]
                voxel_size = data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1]
            except:
                print("Failed to fetch file")
            # Try to get as close as possible to target_size
            
            compression = get_best_size(voxel_size, target_size)
            download_string = f"s{compression}"
            filename = f"{path}{name}.zarr" # !!! with / final cannot be used
            folders_to_ignore = ["masks"]
            if not inference:
                folders_to_ignore.append("inference")
            get_folder_parallel(b, 
                                f"{name}/{name}.zarr/", 
                                filename, 
                                download_string, max_threads, folders_to_ignore)
            # Convert to h5
            zarr_to_h5(filename) 
            print("Converted" + name)
        except Exception as e:
            print(f"Failed to download {name}: {str(e)}.")

def get_all_from_bucket(names, bucket, folder):
    """
    Downloads all files contained in `names` from `bucket` in `folder`.

    Args:
        names (list): array of names of files
        bucket (str): string to the bucket containing the files
        folder (str): path to download folder for the files
    """
    b = q3.Bucket(bucket)
    n_downloaded = 0
    for name in names:
        downloaded = False
        try:
            # Download
            filename = folder + "datasets/" + name + ".zarr" # !!! / is an important difference, not the same as destination_name
            folders_to_ignore = ["masks"]
            get_folder_parallel(b, 
                                f"{name}/{name}.zarr/recon-1/", 
                                filename, 
                                folder, 128, folders_to_ignore)
            downloaded = True
            # Convert
            h5_name = folder + "datasets/" + name + ".h5"
            zarr_to_h5(filename, h5_name)
            n_downloaded += 1
        except Exception as e:
            if downloaded: 
                print(f"Failed to download {name}: {str(e)}.")
            else:
                print(f"Failed to convert {name}: {str(e)}.")
    print(f"Downloaded {n_downloaded} files")

def read_attributes_h5(folder_path, print_path = None):
    """
    Reads the content of all files in the path and prints the voxel size and name in a json file as 
    ```
    name: foo,
    resolution: [z, y, x]
    voxel_size: [z, y, x],
    translation: [z, y, x],
    ```
    Takes as input the path to the folder where the files are stored.
    """
    jsons = []
    sizes = []
    for file in os.listdir(folder_path):
        try:
            with h5py.File(f"{folder_path}{file}", "r") as f:
                # Get group size from /raw-crop
                resolution = f['raw_crop'].shape
                size = list(f.attrs.items())[3][1]

                json_data = {
                    'name': file,
                    'resolution': str(resolution),
                    'voxel_size': str(size),
                    'translation': str(list(f.attrs.items())[4][1])
                }
                print(json_data)
                # Count how many instances of each size 
                found = False
                for i, (s, c) in enumerate(sizes):
                    if (s == size).all():
                        sizes[i] = (s, c + 1)
                        found = True
                if not found:
                    sizes.append((size, 1))
            jsons.append(json_data)
        except Exception as e:
            print(f"Failed to read {file}: {str(e)}")
            continue
    # print("SIZES: ")
    # print(sizes)
    if print_path is not None:
        try:
            with open(print_path, 'w') as outfile:
                json.dump(jsons, outfile)
        except:
            print("Failed to write file")

def scale_input(scale, input_volume, is_segmentation=False):
    """
    @private"""
    if is_segmentation:
        input_volume = rescale(
            input_volume, scale, preserve_range=True, order=0, anti_aliasing=False
        ).astype(input_volume.dtype)
    else:
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
    return input_volume

def resize_to_target(path_to_source, path_to_file, target_size = 8):
    """
    Given the path to a crop, it creates a copy in the destination
    path which approximates the target size. 

    Can perform unlimited downsampling or up to twice upsampling.


    Args:
        path_to_source (str): Path to the source file
        path_to_file (str): Path to the destination file
        target_size (int): Target size in nm between voxels
    """
    # Copy file to preserve metadata and keep the original safe
    try:
        shutil.copyfile(path_to_source, path_to_file)
        file_name = os.path.basename(path_to_file)
        with h5py.File(path_to_file, "r+") as f:
            upsampled_path = None
            # Get downscaling: if -1 or -2, upsample. 
            # If voxel size is over 32, discard crop
            voxel_size = f.attrs["scale"][1]
            if voxel_size == target_size:
                # print("Voxel size is already correct for " + file_name)
                downscaling = 0
                return
            elif voxel_size < target_size: 
                # Case downsampling
                downscaling = get_best_size(voxel_size, target_size)
            elif voxel_size > target_size and voxel_size <= 2*target_size:
                # Case upsampling once (16x)
                downscaling = -1
                upsampled_path = f"{os.path.splitext(path_to_file)[0]}_upsampled{os.path.splitext(path_to_file)[1]}"
            elif voxel_size > 2*target_size and voxel_size <= 4*target_size:
                # Case upsampling twice (32x)
                downscaling = -2
                upsampled_path = f"{os.path.splitext(path_to_file)[0]}_upsampled{os.path.splitext(path_to_file)[1]}"
            else:
                print("Voxel size not supported: " + str(voxel_size) + " for " + file_name)
                os.remove(path_to_file)
                downscaling = None
                return
            # Iterate over the datasets
            # If downsampling, perform it on the spot
            # If upsampling, save in the dictionary for later
            items_dict = {}
            def process_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    # data = f[name]
                    if downscaling > 0:
                        filter_size = 1/(2 ** downscaling)
                        # downsampled_data = block_reduce(data, (filter_size, filter_size, filter_size), np.mean)
                        if name == "raw_crop":
                            downsampled_data = scale_input(3*(filter_size,), data)
                            # rescale, resize
                            obj.resize(downsampled_data.shape)  # Resize the dataset to match the downsampled data
                            obj[...] = downsampled_data
                        else:
                            if name == "label_crop/mito":
                                # Only consider the case mito
                                downsampled_data = scale_input(3*(filter_size,), data, is_segmentation=True)
                                # rescale, resize
                                obj.resize(downsampled_data.shape)  # Resize the dataset to match the downsampled data
                                obj[...] = downsampled_data
                    elif downscaling < 0:
                        # Simpl save everythin to the dictionary
                        items_dict[name] = data
                    else:
                        print("Downscaling not possible. Voxel size: "  + str(voxel_size) + " for " + file_name)
            f.visititems(process_dataset)
            
            # Case upsampling: must create new dataset 
            # Due to HDF5 constraints on dataset size
            if downscaling < 0:
                # Upsample
                # Create second file to be then renamed
                with h5py.File(upsampled_path, 'w') as f2:
                    # Copy attributes
                    new_scale = -2 * downscaling # Only need to handle case -1 and -2 -> 2, 4
                    for item, value in f.attrs.items():
                        f2.attrs[item] = value
                    f2.attrs['scale'] = [x / new_scale for x in f2.attrs['scale']]
                    
                    for label, content in items_dict.items():
                        if label == "raw_crop":
                            f2[label] = scale_input(3*(new_scale,), f[label])
                        elif label == "label_crop/mito":
                            temp = scale_input(3*(new_scale,), f[label], is_segmentation = True)
                            f2[label] = median_filter(temp, size=3)
                        else:
                            continue
                
                # Remove duplicate
                if os.path.exists(path_to_file):
                    os.remove(path_to_file) 
                # Get back to original name
                os.rename(upsampled_path, path_to_file)
                print("Upsampled " , file_name)

            # Update the attributes
            elif downscaling >= 0:
                f.attrs['scale'] = [(x * (2 ** downscaling)) for x in f.attrs['scale']]
                print("Downscaled " + file_name)

    except Exception as e:
        print(f"Failed to resize {path_to_file}: {str(e)}")
        # Delete file if it was created
        upsampled_path = f"{os.path.splitext(path_to_file)[0]}_upsampled{os.path.splitext(path_to_file)[1]}"
        if os.path.exists(path_to_file):
            os.remove(path_to_file)
        if os.path.exists(upsampled_path):
            os.remove(upsampled_path)

def file_check(path_to_file):
    """
    Simply verifies that a file can be opened, gets an attribute and opens the "raw_crop" and "label_crop/mito" dataset.
    """
    try:
        with h5py.File(path_to_file, 'r') as f:
            # Check attrs
            if(f.attrs is not None):
                print(os.path.basename(path_to_file), ": ok, ", str(f.attrs["scale"]))
            else:
                print(os.path.basename(path_to_file), ": failed to get attributes")
            # Check dataset
            try:
                print("First element of dataset: " + str(f["raw_crop"][0][0][0]))
            except Exception as e:
                print(os.path.basename(path_to_file), ": failed to open dataset raw_crop: ", e)
            # Check labels
            try:
                print("First element of labels: " + str(f["label_crop/mito"][0][0][0]))
            except Exception as e:
                print(os.path.basename(path_to_file), ": failed to open dataset mito: ", e)
    except Exception as e:
        print(os.path.basename(path_to_file), ": failed to open + ", e)

def all_to_mito(path_to_source, path_to_file):
    """
    Given the path to a crop, it creates a copy in the destination
    path and attempts to find mitochondria in the "all" label.
    If some is found and mito is not empty, will replace or create 
    a new dataset with the mito inforamtion.

    Args:
        path_to_source (str): Path to the source file
        path_to_file (str): Path where to copy the file
    """
    ids = [3, 4, 5, 50] # IDs that refer to mitochondria
    try:
        with h5py.File(path_to_source, 'r') as f:
            if "label_crop/mito" in f:
                # Case mito is there: check if it is empty
                if np.any(f["label_crop/mito"]):
                    # Case there is something
                    # Skip in any case
                    print("Mito dataset contains mitochondria")
                    shutil.copyfile(path_to_source, path_to_file)
                    return
                # Case it is empty, check all
                if "label_crop/all" in f:
                    list_of_ids = np.unique(f["label_crop/all"])
                    if any(x in list_of_ids for x in ids):
                        # Case there are mitochondria in all and mito is empty
                        # Create a new mito label with 1 if id in ids, 0 otherwise
                        # Create a binary mask
                        binary_mask = np.isin(f["label_crop/all"], ids).astype(np.uint8)
                        
                        # Label connected components in the binary mask
                        labeled_mask, num_features = label(binary_mask)
                        
                        # Assign the labeled mask to new_mito
                        new_mito = labeled_mask.astype(np.uint8)
                        shutil.copyfile(path_to_source, path_to_file)
                        with h5py.File(path_to_file, 'r+') as f2:
                            del f2["label_crop/mito"]
                            f2["label_crop/mito"] = new_mito
                        print("Created copy and added mito from all")
                        return
                    else:
                        # Case mito is empty and there is nothing in all
                        print("Mito is empty and there is no trace in all")
                        return
                else:
                    print("No 'label_crop/all' found. ")
                    return
            else:
                # Case mito is not there
                # same but create 
                if "label_crop/all" in f:
                    list_of_ids = np.unique(f["label_crop/all"])
                    if any(x in list_of_ids for x in ids):
                        # Case there are mitochondria
                        # Create a new mito label with 1 if id in ids, 0 otherwise
                        new_mito = np.isin(f["label_crop/all"], ids).astype(np.uint8)
                        print("Copying: ", path_to_source , " to ", path_to_file)
                        shutil.copyfile(path_to_source, path_to_file)
                        with h5py.File(path_to_file, 'r+') as f2:
                            # Different here
                            f2.create_dataset("label_crop/mito", new_mito.shape, np.uint8, new_mito)
                        print("Created copy and added mito from all")
                        return
                    else:
                        print("No mito and no mito in all found")
                        return
                else:
                    # Case no mito 
                    print("No 'mito' or 'label_crop/all' found. ")
                    return
    except Exception as e:
        print("Failed to get labels: ", e)
    print("Something went wrong for ", path_to_file)


if __name__ == "__main__":
    originals_path = '/scratch-grete/projects/nim00007/data/cellmap/data_crops/'
    dest_path = '/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/mito_crops/'
    file_path = '/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/'
    test_path = f"{file_path}test_crops/"
    blacklist = [
        # Probably corrupted
        "crop_247.h5",
        ]

    for file in os.listdir(dest_path):
        file_check(f"{dest_path}{file}")

    # Testing resizing function
    start = time.time()
    def process_file(file):
        if file not in blacklist:
            print("Processing ", file)
            resize_to_target(f"{originals_path}{file}", f"{dest_path}{file}", 8)

    with ThreadPoolExecutor(max_workers = 64) as executor:
        futures = [executor.submit(process_file, file) for file in os.listdir(originals_path)] 
        for future in as_completed(futures):
            future.result()
    end = time.time()
    print("Getting files took " + str((end-start)) + "s.")