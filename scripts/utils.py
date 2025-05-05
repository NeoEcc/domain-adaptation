# COPILOT
import time
import os
import quilt3 as q3
import json
import h5py
import torch.nn as nn
import torch
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from zarr_utils import zarr_to_h5, export_data
from skimage.measure import block_reduce
from skimage.transform import resize, rescale


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

    # print(f"Best compression for {voxel_list} with tollerance of {tollerance}: {compression}, equivalent to {x*pow(2, compression)} nm")
    return compression

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

def download_folder(bucket, folder_path, download_path):
    """
    Downloads the folder from the folder_path in the bucket to the download_path.
    Required for parallelization
    """
    try:    # Differentiate between folders and files
        name = os.path.basename(folder_path)
        if name == "":
            download_path = download_path + "/"
        # print(f"Downloading {name} to {download_path}")
        bucket.fetch(folder_path, f"{download_path}")
    except Exception as e:
        print(f"Failed to download {folder_path}: {e}")

def read_folder(bucket, path, download_path, folders_to_ignore, name):
    """
    Scans the path for folders, looks for the name:
    - if the name is found, add it to folders_to_download; 
    - otherwise, add all folders to folders_to_explore except those in folders_to_ignore
    return the tuple with the two lists.
    
    Required for parallelization
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
        print("Exploring " + path)
        # Get target folder path
        # Get metadata files .zgroup and .zattrs
        for key in zarr_root_keys:
            if f"{path}{key}" in files:
                folders_to_download.append((f"{path}{key}", f"{download_path}/{key}"))
                # print(f"New file: {download_path}/{key}")
        target = f"{path}{name}/"
        # If the target is there, add it to the list; 
        # Proceed  only otherwise, not to explore other s_ folders
        # print("Looking for " + target + " in " + path + ", with elements:" + str(folders)) 
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

def get_folder_parallel(bucket, path, download_path, name="s0", max_threads=None, folders_to_ignore=[], file_to_read=None):
    """
    Explores folders in buckets and downloads the required folders using as many threads as given.
    Ignores folders with names in the list ["masks", "inference"].

    Args:
        bucket: bucket object where to download from
        path: path inside the bucket (END WITH "/")
        download_path: path where to download the file, of the type base/files/name.zarr CANNOT END WITH "/"
        name: name of the folder to find
        max_threads: number of threads to use for downloading
        folders_to_ignore: list of folders to ignore
        file_to_read: (optional) path to a file with the folders to download, in the format `folder_path \t download_path`
    """
    max_threads = max_threads or os.cpu_count()
    folders_to_explore = [(path, download_path)]
    folders_to_download = []
    start = time.time()
    # The check later will fail if the dataset does not have the required level of downsampling:
    # This will make the search eternally long going through all the folders in s0 and so on
    # An explicit check is required
    other_folders = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
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
        file_path = f"{os.path.dirname(os.path.dirname(download_path))}/files/txt/to_download_{os.path.splitext(os.path.basename(download_path[:-1]))[0]}.txt"
        
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
    print("Finished downloading " + str(len(folders_to_download)) + f"folders in {end - start}s (Found in {mid - start}s).")

def get_some(names, path, bucket_str, inference = False, target_size = 8, max_threads = None):
    """
    Downloads files from a bucket from the `names` list following these steps:
    - Iterate over the list of names
    - Check whether there exists a groundtruth folder (or an inference folder), otherwise skip to next name;
    - Download and read the .zargs file, get the voxel size
    - Decide the downscale level to get closer to the target size
    - Download the correct downscaled version of the dataset
    - Convert to h5

    Parameters:
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

def get_all(names, bucket, folder):
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
    print("SIZES: ")
    print(sizes)
    if print_path is not None:
        try:
            with open(print_path, 'w') as outfile:
                json.dump(jsons, outfile)
        except:
            print("Failed to write file")
    else:
        print("SHAPES:")
        print(jsons)

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

def downsample_to_target(path_to_source, path_to_file, target_size = 8):
    """
    Downsamples the dataset in path to the target size.

    Args:
        path_to_source (str): Path to the source file
        path_to_file (str): Path to the destination file
        target_size (int): Target size in nm between voxels
    """
    # Copy file to preserve metadata # cannot do it anymore, apparently clears metadata
    try:
        shutil.copyfile(path_to_source, path_to_file)

        with h5py.File(path_to_file, "r+") as f:
        
            # Get downscaling: if -1, upsample. If over 16, do nothing
            # print(list(f.attrs.keys()))
            voxel_size = f.attrs["scale"][1]
            if voxel_size == target_size:
                print("Voxel size is already correct for " + path_to_file)
                downscaling = 0
                return
            elif voxel_size < target_size: 
                downscaling = get_best_size(voxel_size, target_size)
            elif voxel_size > target_size and voxel_size <= 2*target_size:
                # print("Voxel size currently not supported: " + str(voxel_size) + " for " + path_to_file)
                downscaling = -1
            elif voxel_size > target_size and voxel_size <= 4* target_size:
                downscaling = -2
            else:
                print("Voxel size currently not supported: " + str(voxel_size) + " for " + path_to_file)
                os.remove(path_to_file)
                downscaling = -100
                return
            print("Voxel size: " + str(voxel_size) + ". Downscaling: " + str(downscaling))
            # Iterate over the datasets
            items_dict = {}
            def process_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if downscaling > 0:
                        # print("Downscaling " + name)
                        filter_size = 1/(2 ** downscaling)
                        # downsampled_data = block_reduce(data, (filter_size, filter_size, filter_size), np.mean)
                        if name == "raw_crop":
                            downsampled_data = scale_input(3*(filter_size,), data)
                            # rescale, resize
                            obj.resize(downsampled_data.shape)  # Resize the dataset to match the downsampled data
                            obj[...] = downsampled_data
                        elif name == "label_crop/mito":
                            downsampled_data = scale_input(3*(filter_size,), data, is_segmentation=True)
                            # rescale, resize
                            obj.resize(downsampled_data.shape)  # Resize the dataset to match the downsampled data
                            obj[...] = downsampled_data
                    elif downscaling == 0:
                        print("No downscaling needed")
                    elif downscaling < 0:
                        # print("Cannot upscale for now")
                        # This should upsample the data, but hdf5 has got a limitation.
                        # The maximum size is defined at data creation and cannot be changed.
                        # Would have to create a new file to make it work, not worth it for 6 elements. Might make a function for that later 
                        # upsampled = resize(data, (data.shape[0]*2, data.shape[1]*2, data.shape[2]*2), mode='reflect', anti_aliasing=True)

                        # Just save stuff to the dictionary
                        items_dict[name] = data
                        
                    else:
                        print("Downscaling not possible. Voxel size: "  + str(voxel_size))
            f.visititems(process_dataset)
            
            if downscaling < 0:
                # Upsample
                with h5py.File(f"{os.path.splitext(path_to_file)[0]}_upsampled{os.path.splitext(path_to_file)[1]}", 'w') as f2:
                    # Copy attributes
                    new_scale = -2 * downscaling # Only need to handle case -1 and -2 -> 2, 4
                    for item, value in f.attrs.items():
                        f2.attrs[item] = value
                    f2.attrs['scale'] = [x / new_scale for x in f2.attrs['scale']]
                    
                    for label, content in items_dict.items():
                        # print("Label: " , label)
                        # print("Item: ", type(content), content.shape)
                        if label == "raw_crop":
                            f2[label] = scale_input(3*(new_scale,), f[label])
                        elif label == "label_crop/mito":
                            f2[label] = scale_input(3*(new_scale,), f[label], is_segmentation = True) 
                        else:
                            continue
                        print("Upsampling ", label)
                        # f2[label] = resize(content, (content.shape[0]*2, content.shape[1]*2, content.shape[2]*2), mode='reflect', anti_aliasing=False)
                
                # Remove duplicate
                if os.path.exists(path_to_file):
                    os.remove(path_to_file) 
                # Get back to original name
                os.rename(f"{os.path.splitext(path_to_file)[0]}_upsampled{os.path.splitext(path_to_file)[1]}", path_to_file)
                print("Upsampled " , path_to_file)

            # Update the attributes
            elif downscaling >= 0:
                print(type(f.attrs['scale'][0]))
                f.attrs['scale'] = [(x * (2 ** downscaling)) for x in f.attrs['scale']]
                
                print("Upsampled " + path_to_file)
    except Exception as e:
        print(f"Failed to downsample {path_to_file}: {str(e)}")
        # Delete file if it was created
        if os.path.exists(path_to_file):
            os.remove(path_to_file)
            

def visualize_some(dataset, number = 4):
    """
    visualize the first `number x number x number` elements of the dataset in a matrix shape
    """ 
    for i in range(number):
        for j in range(number):
            elements = []
            for k in range(number):
                # Get the element at (i, j, k)
                elements.append(dataset[i, j, k])
            print(elements)
        print("")
        print("")

def file_check(path_to_file):
    """
    Simply verifies that a file can be opened, gets an attribute and opens a dataset.
    """
    try:
        with h5py.File(path_to_file) as f:
            # Check attrs
            if(f.attrs is not None):
                print(os.path.basename(path_to_file), ": ok, ", str(f.attrs["scale"]))
            else:
                print(os.path.basename(path_to_file), ": failed to get attributes")
            # Check dataset
            try:
                print("First element of dataset: " + str(f["raw_crop"][0][0][0]))
            except Exception as e:
                print(os.path.basename(path_to_file), ": failed to open dataset + ", e)
            # Check labels
            try:
                print("First element of labels: " + str(f["label_crop/mito"][0][0][0]))
            except Exception as e:
                print(os.path.basename(path_to_file), ": failed to open dataset + ", e)
    except Exception as e:
        print(os.path.basename(path_to_file), ": failed to open + ", e)
# Testsu
if __name__ == "__main__":
    print_path = '/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/txt/'
    originals_path = '/scratch-grete/projects/nim00007/data/cellmap/data_crops/'
    dest_path = '/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/'
    name = 'crop_1'
    blacklist = ["crop_421", ]
    # Check all files
    # for file in os.listdir(dest_path):
    #     file_check(f"{dest_path}{file}")

    # Test upsampling
    for file in ["crop_355.h5"]: # "crop_184.h5", 
        downsample_to_target(f"{originals_path}{file}", f"{dest_path}{file}")    
        # file_check(f"{dest_path}{file}")

    # Test downsampling
    # for file in ["crop_1.h5", "crop_3.h5"]:
    #     downsample_to_target(f"{originals_path}{file}", f"{dest_path}{file}")    
    #     file_check(f"{dest_path}{file}")

    # start = time.time()
    # # On the fly parallelization
    # def process_file(file):
    #     print("Processing ", file)
    #     if file not in blacklist: 
    #         downsample_to_target(f"{originals_path}{file}", f"{dest_path}{file}", 8)

    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_file, file) for file in os.listdir(originals_path)]
    #     for future in as_completed(futures):
    #         future.result()
    # end = time.time()
    # print("Getting files took " + str((end-start)) + "s.")

