# From https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

import h5py
import numpy as np
from glob import glob
import napari
import os

import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt

from torch_em.data import datasets
from torch_em.util.debug import check_loader
from torch_em import default_segmentation_dataset, get_data_loader
from torch_em.transform.label import labels_to_binary, BoundaryTransform
from torch_em.model import UNet3d
from model_utils import *

DATA_DIR = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/datasets/"
LABELS_DIR = ""
RAW_DIR = ""
model = UNet3d(1,1)

# Get all files
volume_paths = glob(os.path.join(DATA_DIR, "*"))

volume = volume_paths[0]
print("The volume extension seems to be:", os.path.splitext(volume_paths[0])[-1])
# with h5py.File(volume) as f:
#     # This fails :/
#     image = f["recon-1/em/fibsem-uint8/s0"][:]  # files/jrc_mus-liver.zarr/recon-1/em/fibsem-uint8/s0
#     label = f["recon-1/labels/groundtruth"][:]  # files/jrc_mus-liver.zarr/recon-1/labels/groundtruth

#     print("The image volume and label volume respectively are:", image.shape, label.shape)
#     plot_samples(image, label, view_napari=True)

# datasets parameters
patch_shape = (1, 512, 512)
volume_paths = glob(os.path.join(DATA_DIR, "*.h5"))  # paths of all the volumes

dataset = default_segmentation_dataset(
    raw_paths=volume_paths,
    raw_key=RAW_DIR,  # this is the hierarchy in the hdf5 files, where the images are stored
    label_paths=volume_paths,
    label_key=LABELS_DIR,   # this is the hierarchy in the hdf5 files, where the labels are stored
    patch_shape=patch_shape,
    label_transform=BoundaryTransform(),  # remember the task above for semantic boundary labels? this function takes care of it.*
    ndim=2,
)