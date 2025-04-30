# From https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

import h5py
import numpy as np
from glob import glob
import napari
import os

import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt
from torch_em.transform.label import labels_to_binary, BoundaryTransform




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
DATA_DIR = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/"

# Instantiate model: 3d, b&w, 54 classes mapped to one mask and one boundary
model = UNet3d(1,2)
# examples to work with a few crops
ex_names = [
    "crop_1.h5",
    "crop_3.h5",
    "crop_4.h5",
    "crop_6.h5",
    "crop_7.h5",
    "crop_8.h5",
    "crop_9.h5",
    "crop_100.h5"
]

ex_paths = [
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_1.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_3.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_4.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_6.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_7.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_8.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_9.h5",
    "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_100.h5",
]

names = os.listdir(DATA_DIR)
        
# Create dataset
dataset = default_segmentation_dataset( # SegmentationDataset
    raw_paths = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_1.h5", #ex_paths,
    raw_key = "raw_crop", 
    patch_shape = (100, 500, 500), 
    label_paths = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/crop_1.h5",
    label_key = "label_crop/mito",   
    ndim=3,
    label_transform=BoundaryTransform()
)
file_path = ex_paths[0]
f = h5py.File(f"{file_path}/raw_crop", "r")
print(f)
# Get statistics about all shapes, figure out the padding soluion 
# Get screw tighter
# print(dataset)

dataloader = get_data_loader(dataset, BATCH_SIZE)
# print(dataloader)

# Trainer:
# torch-em
