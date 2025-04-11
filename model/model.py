# From https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

import h5py
import numpy as np
from glob import glob
import napari
import os

import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt



# import sys
# sys.path.append("/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/torch-em")
from torch_em.data.datasets.electron_microscopy import cellmap

from torch_em.util.debug import check_loader
from torch_em import default_segmentation_dataset, get_data_loader
from torch_em.transform.label import labels_to_binary, BoundaryTransform
from torch_em.model import UNet3d
from torch_em.data import SegmentationDataset
from model_utils import *

# Adapt  torch-em/torch em/data/datasets/electron microscopy/cellmap.py
BATCH_SIZE = 4
DATA_DIR = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/datasets/"
NAMES = [
        "jrc_hela-2",             # 70 GB   # 12 GB     # 36GB in h5??
        "jrc_macrophage-2",       # 96 GB   # 15 GB     # 39GB
        "jrc_jurkat-1",           # 123 GB  # 20 GB     # 44GB
        "jrc_hela-3",             # 133 GB  # 18 GB     # 
        "jrc_ctl-id8-1",          # 235 GB  # ?         # 86G
    ]

# Instantiate model: 3d, b&w, 54 classes so 1->1? 
model = UNet3d(1,1)
# print(model)

# paths = []
# for name in NAMES:
#     paths.append(f"{DATA_DIR}{name}.h5")
paths = f"{DATA_DIR}jrc_hela-2.h5"
print("[PATHS] " + paths)
key = "recon-1/"
# Create dataset
dataset = SegmentationDataset(
    raw_path = paths,
    raw_key = f"{key}em/fibsem-uint8/", # I hope, optional
    patch_shape = (1,128,128),
    label_path = paths,
    label_key = f"{key}labels/",
    ndim=3
)
print(dataset)

# Get dataloader
dataloader = get_data_loader(dataset, BATCH_SIZE, )