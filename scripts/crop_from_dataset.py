import zarr
import os
import numpy as np
import random


from zarr_utils import read_data, export_data

def extract_crops(path_to_zarr, data_key, number_of_crops, crop_size, print_path, blacklist = None):
    """
    From a zarr archive, extract number of crops from a random spot (sequential). 
    Number of crops is per dimension: "2" will produce 4 crops in 2d and 8 in 3d. 
    Returns all crops as an array. Can be saved as hdf5 by another function. 

    Args:
        path_to_zarr (str): The path to the zarr file.
        data_key (str): The key to the data in the zarr archive.
        number_of_crops (int): The number of crops to extract from the zarr archive in each dimension.
        crop_size Tuple(int, int, int): The size of each crop to extract.
        print_path (str): path to the folder where to print the HDF5 crops. 
        blacklist (list, optional): An optional list of crop ROIs to exclude from the extraction. Defaults to None.
    """
    random_slices = []
    extracted_crops = []
    ndim = len(crop_size)
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d tensors are supported")
    n_samples = pow(number_of_crops, ndim)
    # raw = read_data(path_to_zarr + data_key)
    raw = np.arange(1000).reshape((10, 10, 10))
    roi_size = tuple(x * number_of_crops for x in crop_size)
    size = raw.shape
    raw_roi = ( # temp
        slice(0, size[0]),
        slice(0, size[1]),
        slice(0, size[2])
    )
    random_roi = get_random_roi(raw_roi, roi_size, blacklist)
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
                    print(random_slices[-1])
                    init_z += crop_size[2]
            else:
                random_slices.append(
                        (
                            slice(init_x, init_x + crop_size[0]),
                            slice(init_y, init_y + crop_size[1])
                        )
                    )
                print(random_slices[-1])
            init_y += crop_size[1]
        init_x += crop_size[0]
    

def slices_overlap(a, b):
    # Returns True if slices a and b overlap in any dimensions
    x = []
    for d in range(len(a)):
        x.append(a[d].stop < b[d].start or a[d].start >= b[d].stop) 
    return not all(x) # Can be probably made easier with a not and a direct return true

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
    max_attempts = 250

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
    for _ in range(max_attempts):
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
                    break
                
        if not overlap:
            # assert not any(slices_overlap(candidate, bl) for bl in blacklist), f"Overlap detected: {candidate} vs {blacklist}"
            return candidate

    raise RuntimeError(f"Could not find a valid ROI after {max_attempts} attempts")


if __name__ == "__main__":
    n_crops = 2
    test_data_path = "/user/niccolo.eccel/u15001/example_dataset/jrc_ctl-id8-2.zarr"
    data_path = "/scratch-grete/projects/nim00007/data/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr"
    print_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/unlabeled_crops/"
    raw_key = "/recon-1/em/fibsem-uint8/"
    patch_shape = (5,)*3
    # extract_crops(test_data_path, raw_key, n_crops, (3,)*3, print_path)
    data = read_data(test_data_path)
    print(data)