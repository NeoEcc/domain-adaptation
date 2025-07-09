# File dedicated to the functions required to sample crops from a large zarr file and store them for semisupervised training. 

import zarr
import os
import numpy as np
import h5py
import random
import z5py
import h5py
import zarr
import numpy as np
from typing import Tuple
import zarr
import s3fs
import quilt3 as q3

from scipy.ndimage import label

def h5_from_bucket(zarr_path, zarr_key, hdf5_path, roi):
    """
    Given the details in a bucket, downloads only the specified part of a dataset (roi) and
    prints the dataset in an hdf5 file.
    Args:
        zarr_path (str): Path to the zarr file in the S3 bucket.
        zarr_key (str): Key within the zarr file to access the dataset.
        hdf5_path (str): Local path where the HDF5 file will be saved.
        roi (tuple of slices): Region of interest to extract from the dataset.
    """
    fs = s3fs.S3FileSystem(anon=True)
    try:
        # For Zarr files
        store = zarr.storage.FSStore(zarr_path, fs=fs)
        dataset = zarr.open(store, mode='r', path=zarr_key)
        print(f"Successfully opened Zarr dataset: {dataset.shape}, dtype: {dataset.dtype}")
        
        # Extract roi
        roi_data = dataset[roi]
        print(f"Shape: {roi_data.shape}")
        
    except Exception as e:
        print(f"Zarr processing failed: {e}")
        raise
    # Save
    if hdf5_path is not None:
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        with h5py.File(hdf5_path, 'w') as h5_file:
            h5_file.create_dataset("data", data=roi_data, compression="gzip")
            h5_file.attrs["source_path"] = zarr_path
            h5_file.attrs["source_key"] = zarr_key
            h5_file.attrs["roi"] = str(roi)
        print(f"Saved to {hdf5_path}")
    return roi_data

def extract_crops(path_to_zarr, raw_key, number_of_crops, crop_size, raw_roi = None, blacklist = None):
    """
    From a zarr archive, extract number of crops from a random spot (sequential). 
    Number of crops is per dimension: "2" will produce 4 crops in 2d and 8 in 3d. 
    Returns all crops as an array. If print path is given, saves files as hdf5.  

    Args:
        path_to_zarr (str): The path to the zarr file.
        raw_key (str): The key to the data in the zarr archive.
        number_of_crops (int): The number of crops to extract from the zarr archive in each dimension.
        crop_size Tuple(int, int, int): The size of each crop to extract.
        raw_roi (Tuple(slice, slice)): optional ROI within the zarr. If None, will be set as the whole store.
        blacklist (list, optional): An optional list of crop ROIs to exclude from the extraction. Defaults to None.
    """
    
    random_slices = []
    extracted_crops = []
    ndim = len(crop_size)
    # Input checks
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d tensors are supported")
    if raw_roi is not None and len(raw_roi) != ndim:
        raise ValueError("Number of dimensions of roi and crop must match")

    # Read zarr
    zarr_store = zarr.open(path_to_zarr, mode='r')
    raw = zarr_store[raw_key]

    roi_size = tuple(x * number_of_crops for x in crop_size)
    size = raw.shape
    margin = 5
    if raw_roi is None:
        raw_roi = tuple(slice(margin, s - margin) for s in size)
    elif any((raw_roi[s].start > size[s] for s in range(ndim))):
        raise ValueError(f"ROI is outside the crop: {raw_roi} vs {size}")
    elif any((raw_roi[s].stop > size[s] for s in range(ndim))):
        print(f"Warning: ROI is partly outside the crop, could cause issues: {raw_roi} vs {size}")

    random_roi = get_random_roi(raw_roi, roi_size, blacklist)
    # Extract the given number of ROIs from the selected area
    init_x = random_roi[0].start
    for x in range(number_of_crops):
        init_y = random_roi[1].start
        for y in range(number_of_crops):
            if(ndim) == 3:
                init_z = random_roi[2].start
                for z in range(number_of_crops):
                    # print("x, y, z: ", x, ", ",y,", ", z)
                    random_slices.append(
                        (
                            slice(init_x, init_x + crop_size[0]),
                            slice(init_y, init_y + crop_size[1]),
                            slice(init_z, init_z + crop_size[2])
                        )
                    )
                    init_z += crop_size[2]
            else:
                random_slices.append(
                        (
                            slice(init_x, init_x + crop_size[0]),
                            slice(init_y, init_y + crop_size[1])
                        )
                    )
            init_y += crop_size[1]
        init_x += crop_size[0]
    # Get the crops from zarr
    for ex_slice in random_slices:
        np_slice = np.s_[ex_slice]
        extracted_crop = (raw[np_slice], ex_slice)
        extracted_crops.append(extracted_crop)
        
    return extracted_crops
    
def extract_samples(path_to_zarr, data_key, crop_number, crops_per_batch_dim, crop_size, print_path, raw_roi = None, blacklist = None):
    """
    From a zarr archive, extract number of crops from a set of random spots and saves them as HDF5. 
    Number of crops is per dimension: "2" will produce 4 crops in 2d and 8 in 3d. 

    Args:
        path_to_zarr (str): The path to the zarr file.
        data_key (str): The key to the data in the zarr archive.
        crop_number (int): number of crops to be extracted, total. Must be a multiple of `crops_per_batch_dim^dim`
        crops_per_batch_dim (int): Per-dimension batch size to extract from each random point. 
        crop_size Tuple(int, int, int): The size of each crop to extract.
        print_path (str): path to the folder where to print the HDF5 crops; None to skip this step.
        raw_roi (Tuple(slice, slice)): optional ROI within the zarr. If None, will be set as the whole store.
        blacklist (list, optional): An optional list of crop ROIs to exclude from the extraction. Defaults to None.
    """
    # Input checks
    ndim = len(crop_size)
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d tensors are supported")
    if raw_roi is not None and len(raw_roi) != ndim:
        raise ValueError("Number of dimensions of roi and crop must match")
    if print_path is None:
        raise ValueError("print_path cannot be None")
    elif print_path[-1] != "/":
        print_path += "/"
    # Batch check
    magic_number = pow(crops_per_batch_dim, ndim)
    if crop_number % magic_number != 0:
        print(f"Warning: {crop_number} is not a multiple of {magic_number}.")
        crop_number = crop_number - crop_number % magic_number + magic_number
        print(f"Will generate {crop_number} samples instead. ")
    # Crop sampling and writing 
    offset = 0 # In case of creating many times, to get different names. 
    generated_crops = 0 + offset
    while generated_crops < crop_number + offset:
        print("Getting new batch")
        crops = extract_crops(path_to_zarr, data_key, crops_per_batch_dim, crop_size, raw_roi = raw_roi, blacklist = blacklist)
        for crop, coordinates in crops:
            generated_crops += 1
            # To store coordinates as an attribute
            starts = tuple(s.start for s in coordinates)
            stops  = tuple(s.stop for s in coordinates)
            jsonable_coordinates = (starts, stops)

            with h5py.File(f"{print_path}raw_crop_{generated_crops}.h5", "w") as f:
                f.create_dataset("raw", data=crop.astype(crop.dtype))
                f.attrs["voxel_size"] = (8,8,8)
                f.attrs["coordinates"] = jsonable_coordinates
                f.attrs["source_dataset"] = "https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/"
        print("Batch created. Current crops created: ", generated_crops)

def extract_labeled_sample(path_to_dataset, raw_roi, raw_key, label_key, print_path, path_to_labels = None, label_roi = None):
    """
    Given the path to a zarr archive, a ROI within it, and the path to a label file,
    creates an HDF5 file containing the raw crop and the labels. 

    Args:   
        path_to_raw (str): path to the zarr archive with the raw.  
        raw_roi (Tuple(Slice(int, int), ...)): tuple of slices representing the position
        of the crop in the raw dataset.
        label_key (str): position of the labels in the file. 
        raw_key (str): position of the raw in the file.
        print_path (str): path to where the file will be printed, including filename.
        path_to_labels (str): path to the zarr archive with the labels. 
            If None, will be inferred as the same as the raw or from the origin
        label_roi: if given, will be used. If None, will be considered from the origin the same size as the crop. 

    """
    # Initial checks
    if raw_roi is None:
        raise ValueError("ROI must be given")
    ndim = len(raw_roi)
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d tensors are supported")
    if print_path is None:
        raise ValueError("print_path cannot be None")
    if path_to_labels is None:
        path_to_labels = path_to_dataset

    # Open zarr, get raw and labels
    raw_zarr_store = zarr.open(path_to_dataset, mode='r')
    label_zarr_store = zarr.open(path_to_labels, mode='r')
    raw_dataset = raw_zarr_store[raw_key]
    label_dataset = label_zarr_store[label_key]


    # get slices of raw, calculate slice for label if None
    np_slice = np.s_[raw_roi]
    print("np_slice: ", np_slice)
    raw_crop = raw_dataset[np_slice]
    print("slice:         ", np_slice)
    print("raw size:      ", raw_dataset.shape)

    print("label_dataset: ", label_dataset.shape)
    print("raw_crop:      ", raw_crop.shape)
    print("Raw roi:       ", raw_roi)

    # If the same ROI as raw is too large, adapt to be 0 to size 
    size = raw_crop.shape
    if any((label_roi[s].stop > size[s] for s in range(ndim))):
        label_roi = tuple(slice(0, s) for s in size)
    else:
        label_roi = raw_roi

    # Create labels and separate instances
    mito, _ = label(label_dataset[label_roi])
    if not mito.any():
        print("Warining: empty labels found")
    # Print to file
    with h5py.File(print_path, "w") as f:
        f.create_dataset("raw_crop", data = raw_crop)
        f.create_dataset("label_crop/mito", data = mito)
        f.attrs["voxel_size"] = (8,8,8)
        f.attrs["source_dataset"] = "https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/"

def get_sub_crops(original_crop, crop_size):
    """
    From a large crop, extracts as many smaller crops as possible in an array.

    Args:
        original_crop (np.array): array from which to get the crops
        crop_size (Tuple(int)): size of the sub-crops
    """   
    # Input checks
    ndim = len(crop_size)
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d tensors are supported")
    if ndim != len(original_crop.shape):
        raise ValueError("Dimensions of crop must be the same as the slice")
    if any(s < c for s, c in zip(original_crop.shape, crop_size)): 
        raise ValueError(f"Inserted crop is smaller than the slice size ({original_crop.shape},{crop_size})")
    elif all(s == c for s, c in zip(original_crop.shape, crop_size)):
        print("Warining: size of original crop and crop size are the same")
        return [original_crop]
    elif any(s % c != 0 for s, c in zip(original_crop.shape, crop_size)):
        print("Warining: part of the crop will be discarded. ", crop_size, " is not a multiple of ", original_crop.shape)
    sub_crops = []
    slices = []

    # Per-dimension division, can iterate indefinitely since the checks have been made
    x = 0
    while x <= original_crop.shape[0] - crop_size[0]:
        y = 0
        while y <= original_crop.shape[1] - crop_size[1]:
            temp_slice = (
                slice(x, x + crop_size[0]),
                slice(y, y + crop_size[1])
            )
            if ndim == 3:
                z = 0
                while z <= original_crop.shape[2] - crop_size[2]:
                    # print("x,y,z: ",  x, " ",y , " ", z)
                    temp_slice_z = temp_slice + (slice(z, z + crop_size[2]),)
                    slices.append(temp_slice_z)
                    z += crop_size[2]
            else:
                slices.append(temp_slice)
            y += crop_size[1]
        x += crop_size[0]
    # Get crops from slices
    for x in slices:
        np_slice = np.s_[x]
        sub_crop = original_crop[np_slice]
        sub_crops.append(sub_crop)
    return sub_crops
    # # To get crops from a file:
    # with h5py.File("/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/unlabeled_crops/raw_crop_1.h5", "r") as f:
    #     raw = f["raw"]
    #     coordinates = f.attrs["coordinates"]
    #     sub_crops = get_sub_crops(raw, (128,)*3)
    # x = 1
    # for crop in sub_crops:
    #     with h5py.File(f"/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/unlabeled_crops/raw_crop_1_{x}.h5", "w") as f:
    #         f.create_dataset("raw", data=crop.astype(crop.dtype))
    #         f.attrs["voxel_size"] = (8,8,8)
    #         f.attrs["approx_coordinates"] = coordinates
    #         f.attrs["source_dataset"] = "https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/"
    #     print("created ", x)
    #     x += 1

def slices_overlap(a, b):
    """
    Returns True if the hyperrectangles defined by tuples of slices `a` and `b` overlap
    in all dimensions
    """
    return all(s1.start < s2.stop and s2.start < s1.stop for s1, s2 in zip(a, b))

def get_random_roi(original_roi, size, blacklist = None):
    """
    Returns a random region of interest within the original roi coordinate space
    that does not include any voxel in the blacklist, of the specified size. 
    Works in any dimension. Boundaries of the blacklist are also excluded.

    Algorithm, to be used dimension-wide:
    - Get a starting point between beginning and end - size
    - Create final point adding size to starting point
    - For all items in blacklist:
        - If the initial point of the roi is larger than the final point, all good
        - If the final point is smaller than the initial point, all good.
    - If these conditions are met, return a valid roi.
    - Otherwise, add 1 to the counter and restart. 
    - If the counter reaches a threshold, stop. Could be extremely unlucky or wrong initialization.
        
    Args:
        original_roi: The original region of interest within which to sample.
        size: The shape (tuple of ints) of the desired random ROI.
        blacklist: A list of regions to avoid when sampling the ROI. None to accept all samples. 
        Must be an array of tuples of slices.
    """
    max_attempts = 1000

    # Input checks
    if original_roi is None:
        raise ValueError("original_roi must be provided")
    if isinstance(original_roi, tuple):
        roi_slices = original_roi
    else:
        roi_slices = tuple(original_roi)

    # If there is some none, set to 0 - makes sense only in the start
    starts = [slc.start if slc.start is not None else 0 for slc in roi_slices]
    stops = [slc.stop for slc in roi_slices]
    dims = len(starts)

    # Get the maximum starting point between start and end - size
    valid_starts = []
    for d in range(dims):
        max_start = stops[d] - size[d]
        if max_start < starts[d]:
            raise ValueError(f"ROI size {size} is too large for dimension {d} in original_roi {original_roi}")
        valid_starts.append((starts[d], max_start))

    # Try up to max_attempts times to find a non-blacklisted ROI
    for n in range(max_attempts):
        rand_start = [random.randint(valid_starts[d][0], valid_starts[d][1]) for d in range(dims)]
        rand_stop = [rand_start[d] + size[d] for d in range(dims)]
        candidate = tuple(slice(rand_start[d], rand_stop[d]) for d in range(dims))

        # Check overlap
        overlap = False
        if blacklist is not None:
            for bl in blacklist:
                # print("Checking ", candidate, " and ", bl)
                if slices_overlap(candidate, bl):
                    overlap = True
                    if n >= (max_attempts -3):
                        print(f"{n} attempts used - blacklisted item: ", bl, " for ", original_roi) 
                    break
                
        if not overlap:
            return candidate

    raise RuntimeError(f"Could not find a valid ROI after {max_attempts} attempts for {size} in {original_roi}")

def n5_to_hdf5(n5_path, hdf5_path, roi = None):
    """Convert N5 file to HDF5 with optional roi
        Works only with single arrays
    """
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
                    with h5py.File(hdf5_path, 'w') as h5_file:
                        h5_file.create_dataset(ds_name, data=data, compression="gzip")
                
            print(f"Successfully converted to HDF5: {hdf5_path}")
            return data
        except Exception as e2:
            print("Failed to convert twice: ", e, e2)
            raise e2

if __name__ == "__main__":
    # Create test ROI
    size = (1000,)*3  # Smaller size for testing
    train_roi = (  # Divide by 8 as coordinates are in nm and 1 px = 8nm
        slice(35000/8, 50000/8), # x
        slice(20000/8, 50000/8), # y
        slice(30000/8, 80000/8), # z
    )
    selected_roi_1 = (   # 80825, 13486, 67009 - 66418, 32242, 56951
        slice(6687, 8250, None), 
        slice(1850, 4030, None), 
        slice(8375, 10125, None),
    )
    selected_roi_2 = ( #  75397, 69836, 980 - 100677, 61274, 17697
        slice(250, 1875, None),
        slice(7375, 8912, None), 
        slice(10000, 11875, None), 
    )
    full_test_roi = (
        slice(28000/8, 35000/8), # x
        slice(20000/8, 50000/8), # y
        slice(30000/8, 80000/8), # z
    )
    test_roi = (
        slice(30000//8, 30000//8 + size[0]),  # x
        slice(30000//8, 30000//8 + size[1]),  # y
        slice(55000//8, 55000//8 + size[2]),  # z
    )
    # Check that test has never been used
    assert not slices_overlap(test_roi, train_roi)
    assert not slices_overlap(test_roi, selected_roi_1)
    assert not slices_overlap(test_roi, selected_roi_2)
    assert slices_overlap(test_roi, full_test_roi)

    print("zarr-mito_seg")
    h5_from_bucket("janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr", "recon-1/labels/masks/evaluation", "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/test_labels_L.h5", test_roi)
    print("zarr-raw")
    h5_from_bucket("janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr", "recon-1/em/fibsem-uint8/s0", "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/test_crop_L.h5", test_roi)

    print("n5-mito_seg")
    h5_from_bucket("janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5", "labels/mito_seg/s0", None, test_roi)
    print("n5-mito_bag_seg")
    h5_from_bucket("janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5", "labels/ariadne/mito_bag_seg", None, test_roi)

    # n_crops = 1
    # n_crops_dim = 1 # !!! 2 -> 8, 3 -> 27, 4 -> 64   

    # data_path = "/scratch-grete/projects/nim00007/data/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr"
    # print_path_unlabeled = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/unlabeled_crops/"
    # base_path_labels = "/user/niccolo.eccel/u15001/example_dataset/"
    # raw_key = "/recon-1/em/fibsem-uint8/s0/"
    # label_key = "/mito/s0/"
    # patch_shape = (1024,)*3
    

    blacklist = [
        (   # crop 136         ("crop136", ([4424, 3507, 4562], [4624, 3707, 4762])), zyx
            slice(4562, 4624),
            slice(3507, 3707),
            slice(4424, 4762),
        ),
        (   # crop 144         ("crop144", ([5006, 6090, 5704], [5106, 6190, 5804])), zyx
            slice(5704, 5804), 
            slice(6090, 6090),
            slice(5006, 5106),
        ),
        (   # crop 124         ("crop124", ([43796, 36356, 76836], [45396, 37956, 78436])),
            slice(9604, 9804),
            slice(4544, 4744),
            slice(5474, 5674),
        ),
        (   # crop 135         ("crop135", ([34620, 39036, 54876], [36220, 40636, 56476])),
            slice(6859, 7059),
            slice(4879, 5079),
            slice(4327, 4527),
        ),
        (   # crop 139         ("crop139", ([36668, 46420, 72996], [37468, 47220, 73796])),
            slice(9124, 9224),
            slice(5802, 5902),
            slice(4583, 4683),
        ),
        (   # crop 142         ("crop142", ([34804, 41124, 65884], [35604, 41924, 66684])),
            slice(8235, 8335),
            slice(5140, 5240),
            slice(4350, 4450),
        ),
    ]
    for b in blacklist:
        assert not slices_overlap(test_roi, b), "Ok"


    # data_voxel = [ # !!! Z Y X
    #     ("crop124", ([5474, 4544, 9604], [5674, 4744, 9804])),
    #     ("crop125", ([4015, 10412, 7608], [4215, 10612, 7808])),
    #     ("crop131", ([2269, 2912, 2964], [2319, 3112, 3164])),
    #     ("crop132", ([4442, 5715, 3321], [4642, 5915, 3521])),
    #     ("crop133", ([4274, 7349, 5699], [4474, 7549, 5899])),
    #     ("crop135", ([4327, 4879, 6859], [4527, 5079, 7059])), 
    #     ("crop136", ([4424, 3507, 4562], [4624, 3707, 4762])),
    #     ("crop137", ([6999, 4150, 9016], [7199, 4350, 9216])),
    #     ("crop138", ([7999, 4599, 6499], [8199, 4799, 6699])),
    #     ("crop139", ([4583, 5802, 9124], [4683, 5902, 9224])), 
    #     ("crop142", ([4350, 5140, 8235], [4450, 5240, 8335])),
    #     ("crop143", ([2634, 4952, 6474], [2734, 5052, 6574])),
    #     ("crop144", ([5006, 6090, 5704], [5106, 6190, 5804])),
    #     ("crop145", ([2194, 7718, 5793], [2374, 8093, 6168])),
    #     ("crop150", ([4348, 2024, 9124], [4448, 2124, 9224])),
    #     ("crop151", ([1696, 5187, 5592], [1796, 5287, 5692])),
    #     ("crop157", ([8654, 4774, 2999], [8754, 4874, 3099])),
    #     ("crop171", ([4468, 6596, 6780], [4668, 6658, 6980])),
    #     ("crop172", ([2038, 7178, 4156], [2288, 7428, 4356])),
    #     ("crop175", ([2042, 7252, 4619], [2442, 7402, 4919])),
    #     ("crop177", ([799, 399, 399], [1599, 1199, 1199])),
    #     ("crop183", ([8654, 4774, 2999], [8754, 4874, 3099])),
    #     ("crop416", ([4468, 6596, 6780], [4668, 6658, 6980])),
    #     ("crop417", ([2038, 7178, 4156], [2288, 7428, 4356])),
    # ]

    # name, ([z, y, x], [zz, yy, xx]) = data_voxel[-3]
    # raw_roi = (
    #     slice(z, zz),
    #     slice(y, yy),
    #     slice(x, xx),
    # )

    # label_key = ""
    # test_crop_path = f"{base_path_labels}mito_instance_seg.zarr"
    # print_test = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/test_crop.h5"
    # n5_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/temp_inference.n5"
    # # extract_labeled_sample(data_path, test_roi, raw_key, label_key, print_test, n5_path)

    # import quilt3 as q3
    # b = q3.Bucket("s3://janelia-cosem-datasets")
    # b.fetch("jrc_mus-liver/jrc_mus-liver.n5/labels/ariadne/mito_instance_seg/", "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/temp_instance_seg.n5/")
    # b.fetch("jrc_mus-liver/jrc_mus-liver.zarr/recon-1/labels/masks/evaluation/", "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/temp_evaluation.zarr/")