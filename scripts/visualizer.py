# # import time
# from typing import Any, Dict
# # from config import *
# import h5py
# import zarr
# import argparse
# import os
# from glob import glob
# import numpy as np
# import napari
# # import elf.parallel as parallel
# from elf.io import open_file
# from tqdm import tqdm
# import z5py
# from tifffile import imread


# def get_file_paths(path, ext=".h5", reverse=False):
#     if ext in path:
#         return [path]
#     else:
#         paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
#         return paths


# def visualize_data(data):
#     viewer = napari.Viewer()
#     for key, value in data.items():
#         # Handle NaN values by replacing them with zeros for all data types
#         value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Skip data with zero dimensions
#         if value.size == 0 or any(dim == 0 for dim in value.shape):
#             print(f"Skipping {key} due to zero dimensions: {value.shape}")
#             continue
        
#         # Convert to appropriate data type for labels
#         if not (key == "raw" or "raw" in key or key == "prediction" or "pred" in key or "foreground" in key or "boundary" in key):
#             # For labels, ensure we have integer type
#             if not np.issubdtype(value.dtype, np.integer):
#                 value = value.astype(np.int32)
                
        
#         if key == "raw" or "raw" in key:
#             print("Added raw: ", key)

#             viewer.add_image(value, name=key)
#         elif key == "prediction" or "pred" in key or "foreground" in key or "boundary" in key or "foreground" in key:
#             print("Added label (1): ", key)
#             # Apply threshold: set values >= 0.9 to 1, else 0
#             value = (value >= 0.9).astype(np.int32)
#             viewer.add_labels(value, name=key, blending="additive")
#         else:
#             viewer.add_labels(value, name=key)
#             print("Added label (2): ", key)

#     # Get the "raw" layer
#     raw_layer = next((layer for layer in viewer.layers if "raw" in layer.name), None)
#     if raw_layer:
#         # Remove the "raw" layer from its current position
#         viewer.layers.remove(raw_layer)
#         # Add the "raw" layer to the beginning of the layer list
#         viewer.layers.insert(0, raw_layer)

#     napari.run()

# skip = [
#     "ecs", "pm", "golgi_mem", "golgi_lum", "mito_mem", "mito_lum", "mito_ribo", 
#     "ves_mem", "ves_lum", "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
#     "ld_mem", "ld_lum", "er_mem", "er_lum", "eres_mem", "eres_lum", "ne_mem",
#     "ne_lum", "np_out", "np_in", "hchrom", "nhchrom", "echrom", "nechrom",
#     "nucpl", "nucleo", "mt_out", "cent", "cent_dapp", "cent_sdapp", "ribo",
#     "cyto", "mt_in", "nuc", "vim", "glyco", "golgi", "ves", "endo", "lyso",
#     "ld", "rbc", "eres", "perox_mem", "perox_lum", "perox", "er", #"mito", 
#     "ne", "np", "chrom", "mt", "cell", "actin", "tbar", "bm", "er_mem_all",
#     "ne_mem_all", "cent_all", "isg_ins", "isg_mem", "all", "isg", "isg_lum",
#     # TEMP
#     "boundary",
#     "mito",
#     # "foreground",
#     # "raw_crop",
#     # "label_crop",
#     "foreground_unprocessed",
# ]

# def should_skip(key_name: str, skip_list: list) -> bool:
#     """
#     Check if a key should be skipped based on the skip list.
#     This checks for exact matches and partial matches.
#     """
#     # Check exact match
#     if key_name in skip_list:
#         return True
#     # for skip_item in skip_list:
#     #     if skip_item in key_name:
#     #         return True
    
#     # Check if the key name (without path) matches any skip item
#     base_key = key_name.split('/')[-1]  # Get the last part after '/'
#     if base_key in skip_list:
#         return True
        
#     return False

# def extract_data(group: Any, data: Dict[str, Any], prefix: str = "", scale: int = 1):
#     """
#     Recursively extract datasets from a group and store them in a dictionary.
#     """

#     for key, item in group.items():
#         full_key = f"{prefix}/{key}" if prefix else key
        
#         # Check if this key should be skipped
#         if should_skip(key, skip) or should_skip(full_key, skip):
#             print(f"Skipping: {full_key}")
#             continue
            
#         if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
#             # Recursively extract data from subgroups
#             extract_data(item, data, prefix=full_key, scale=scale)
#         else:
#             ndim = item.ndim
#             # Generate a slicing tuple based on the number of dimensions
#             slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

#             # Apply downsampling while preserving batch/channel dimensions
#             extracted_data = item[slicing] if scale > 1 else item[:]
#             # Handle NaN values immediately after extraction
#             data[full_key] = np.nan_to_num(extracted_data, nan=0.0, posinf=0.0, neginf=0.0)
#             print(f"Loaded: {full_key}")
#             # # Store the dataset in the dictionary
#             # data[full_key] = item[:]



# def main(root_path: str, ext: str = None, scale: int = 1, upsample: bool = False, root_label_path: str = None):
#     if ext is None:
#         print("Loading h5, n5 and zarr files")
#         paths = get_file_paths(root_path, ".h5")
#         paths.extend(get_file_paths(root_path, ".n5"))
#         paths.extend(get_file_paths(root_path, ".zarr"))
#         paths.extend(get_file_paths(root_path, ".mrc"))
#         paths.extend(get_file_paths(root_path, ".rec"))
#     else:
#         paths = get_file_paths(root_path, ext)
#     if root_label_path is not None:
#         label_paths = get_file_paths(root_label_path, ".tif")
#     else:
#         label_paths = None
#     print("Found files:", len(paths))
#     for path in tqdm(paths):
#         print("\n", path)
#         label_path = None
#         # if label_paths is not None:
#         #     label_path = util.find_label_file(path, label_paths)
#         # else:
#         #     label_path = None
#         with open_file(path, mode="r") as f:
#             data = {}
#             if label_path is not None:
#                 print("Loading label data from", label_path)
#                 if "data" in f.keys():
#                     ndim = f["data"].ndim
#                 elif "raw" in f.keys():
#                     ndim = f["raw"].ndim
#                 else:
#                     print("Warning! Assuming NDIM = 3")
#                     ndim = 3
#                 slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
#                 label_data = imread(label_path)[slicing] if scale > 1 else imread(label_path)
#                 data["label"] = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)
#             else:
#                 print("No specific label path loaded.")
#             if ".mrc" in path or ".rec" in path:
#                 ndim = f["data"].ndim
#                 slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
#                 extracted_data = f["data"][slicing] if scale > 1 else f["data"][:]
#                 data["raw"] = np.nan_to_num(extracted_data, nan=0.0, posinf=0.0, neginf=0.0)
#             else:
#                 print(f.keys())
#                 for key in f.keys():
#                     # Check if this top-level key should be skipped
#                     if should_skip(key, skip):
#                         print(f"Skipping key: {key}")
#                         continue
                        
#                     if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
#                         print(f"Loading group: {key}")
#                         extract_data(f[key], data, scale=scale)
#                         continue
#                     ndim = f[key].ndim

#                     # Generate a slicing tuple based on the number of dimensions
#                     slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
#                     # Apply downsampling while preserving batch/channel dimensions
#                     extracted_data = f[key][slicing] if scale > 1 else f[key][:]
#                     data[key] = np.nan_to_num(extracted_data, nan=0.0, posinf=0.0, neginf=0.0)
#                     print(f"Loaded top-level: {key}")

#         # if upsample:
#         #     del data["pred"]
#         #     del data["raw"]

#         #     for key in data.keys():
#         #         data[key] = upsample_data(data[key], upsample)

#         raw_shape = None
#         for k in data.keys():
#             if "raw" in k:
#                 raw_shape = data[k].shape
        
#         visualize_data(data)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", "-p", type=str)
#     parser.add_argument("--ext", "-e", type=str, default=None)
#     parser.add_argument("--scale", "-s", type=int, default=1)
#     parser.add_argument("--upsample", "-u", type=int, default=None)
#     parser.add_argument("--label_path", "-lp", type=str, default=None)
#     args = parser.parse_args()
#     path = args.path
#     ext = args.ext
#     scale = args.scale
#     upsample = args.upsample
#     label_path = args.label_path
#     main(path, ext, scale, upsample, label_path)



# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--path", "-p", type=str)
# #     args = parser.parse_args()
# #     path = args.path
# #     # path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops/crop_325.h5"
# #     viewer = napari.Viewer()
# #     with h5py.File(path) as f:
# #         for key, value in f.items():
# #             if key in ["label", "label_crop"]:
# #                 viewer.add_labels(f[key])
# #             else:
# #                 viewer.add_image(f[key])
# #     napari.run()










# From synapse-net

from typing import Any, Dict
# from config import *
import h5py
import zarr
import argparse
import os
from glob import glob
# import numpy as np
import napari
# import elf.parallel as parallel
from elf.io import open_file
from tqdm import tqdm
import z5py
from tifffile import imread


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths

def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if key == "raw" or "raw" in key or "foreground" in key or "boundary" in key:
            print(key)
            # if data[key].ndim == 4:
            #     data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
            # else:
            #     value = torch_em.transform.raw.normalize_percentile(value, lower=1, upper=99)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_labels(value, name=key)
    # Get the "raw" layer
    raw_layer = next((layer for layer in viewer.layers if "raw" in layer.name), None)
    if raw_layer:
        # Remove the "raw" layer from its current position
        viewer.layers.remove(raw_layer)
        # Add the "raw" layer to the beginning of the layer list
        viewer.layers.insert(0, raw_layer)

    napari.run()

def extract_data(group: Any, data: Dict[str, Any], prefix: str = "", scale: int = 1):
    """
    Recursively extract datasets from a group and store them in a dictionary.
    """
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            # Recursively extract data from subgroups
            extract_data(item, data, prefix=full_key, scale=scale)
        else:
            ndim = item.ndim
            # Generate a slicing tuple based on the number of dimensions
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

            # Apply downsampling while preserving batch/channel dimensions
            data[full_key] = item[slicing] if scale > 1 else item[:]
            # # Store the dataset in the dictionary
            # data[full_key] = item[:]

def main(root_path: str, ext: str = None, scale: int = 1, upsample: bool = False, root_label_path: str = None):
    if ext is None:
        print("Loading h5, n5 and zarr files")
        paths = get_file_paths(root_path, ".h5")
        paths.extend(get_file_paths(root_path, ".n5"))
        paths.extend(get_file_paths(root_path, ".zarr"))
        paths.extend(get_file_paths(root_path, ".mrc"))
        paths.extend(get_file_paths(root_path, ".rec"))
    else:
        paths = get_file_paths(root_path, ext)
    print("Found files:", len(paths))
    for path in tqdm(paths):
        print("\n", path)
        label_path = None
        # if label_paths is not None:
        #     label_path = util.find_label_file(path, label_paths)
        # else:
        #     label_path = None
        with open_file(path, mode="r") as f:
            data = {}
            if label_path is not None:
                print("Loading label data from", label_path)
                if "data" in f.keys():
                    ndim = f["data"].ndim
                elif "raw" in f.keys():
                    ndim = f["raw"].ndim
                else:
                    print("Warning! Assuming NDIM = 3")
                    ndim = 3
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["label"] = imread(label_path)[slicing] if scale > 1 else imread(label_path)
            else:
                print("No specific label path loaded.")
            if ".mrc" in path or ".rec" in path:
                ndim = f["data"].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
            else:
                print(f.keys())
                for key in f.keys():
                    if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                        print(f"Loading group: {key}")
                        extract_data(f[key], data, scale=scale)
                        continue
                    ndim = f[key].ndim

                    # Generate a slicing tuple based on the number of dimensions
                    slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                    # Apply downsampling while preserving batch/channel dimensions
                    data[key] = f[key][slicing] if scale > 1 else f[key][:]
        
        visualize_data(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--scale", "-s", type=int, default=1)
    parser.add_argument("--upsample", "-u", type=int, default=None)
    parser.add_argument("--label_path", "-lp", type=str, default=None)
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    scale = args.scale
    upsample = args.upsample
    label_path = args.label_path
    main(path, ext, scale, upsample, label_path)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", "-p", type=str)
    # args = parser.parse_args()
    # path = args.path
    # # path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops/crop_325.h5"
    # viewer = napari.Viewer()
    # with h5py.File(path) as f:
    #     for key, value in f.items():
    #         if key in ["label", "label_crop"]:
    #             viewer.add_labels(value)
    #         else:
    #             viewer.add_image(value)
    # napari.run()