# TODO:
#  Import old model
#  Import parameters: patch size, 
#  Define new parameters?
#  Define new paths
#  Create new dataloader
#    Find out the correct number of samples, probably 10x the number of pixel (avg 165x) or 10x the number of samples
#    134 * 10 * 165^3 / 128^3 = 2870 samples
#  Train 
import os

from torch_em.util import load_model
from synapsenet.semisupervised_utils import semisupervised_training

from UNet import model

# Name of the model to be used as a base and the new model
old_model_name = "Anisotropic-3d-UNet-128-1"
new_model_name = f"{old_model_name}-DA"

# Patch shape of the model
patch_shape = (128,)*3

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the data in the target domain
target_data_path = "/scratch-grete/projects/nim00007/data/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr"
raw_key = "/recon-1/em/fibsem-uint8/s0/"
label_key = "/recon-1/labels/groundtruth/crop000/"
raw_roi = (
    slice(30000, 80000),
    slice(20000, 50000),
    slice(35000, 50000)
)

# Path to the best version of the previous model
best_path = f"{save_path}checkpoints/{old_model_name}/best.pt"

# Load model for training
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

model = load_model(best_path, model)
model.train()

if __name__ == "__main__":
    semisupervised_training(
        name = old_model_name, model = model, unlabeled_train_paths = target_data_path, val_paths = (target_data_path), 
        label_key = "", patch_shape = patch_shape, save_root = save_path
    )

# - ROI for mitochondria example:
# 	- x [30000:80000]
# 	- y [20000:50000]
# 	- z [35000:50000]

# ROI of new samples:

    data = [
    ("crop124", ([10949, 9089, 19209], [11349, 9489, 19609])),
    ("crop125", ([8031, 20825, 15217], [8431, 21225, 15617])),
    ("crop131", ([4539, 5825, 5929], [4639, 6225, 6329])),
    ("crop132", ([8885, 11431, 6643], [9285, 11831, 7043])),
    ("crop133", ([8549, 14699, 11399], [8949, 15099, 11799])),
    ("crop135", ([8655, 9759, 13719], [9055, 10159, 14119])),
    ("crop136", ([8849, 7015, 9125], [9249, 7415, 9525])),
    ("crop137", ([13999, 8301, 18033], [14399, 8701, 18433])),
    ("crop138", ([15999, 9199, 12999], [16399, 9599, 13399])),
    ("crop139", ([9167, 11605, 18249], [9367, 11805, 18449])),
    ("crop142", ([8701, 10281, 16471], [8901, 10481, 16671])),
    ("crop143", ([5269, 9905, 12949], [5469, 10105, 13149])),
    ("crop144", ([10013, 12181, 11409], [10213, 12381, 11609])),
    ("crop145", ([4389, 15437, 11587], [4749, 16187, 12337])),
    ("crop150", ([8697, 4049, 18249], [8897, 4249, 18449])),
    ("crop151", ([3393, 10375, 11185], [3593, 10575, 11385])),
    ("crop157", ([17309, 9549, 5999], [17509, 9749, 6199])),
    ("crop171", ([8937, 13193, 13561], [9337, 13317, 13961])),
    ("crop172", ([4077, 14357, 8313], [4577, 14857, 8713])),
    ("crop175", ([4085, 14505, 9239], [4885, 14805, 9839])),
    ("crop177", ([1599, 799, 799], [3199, 2399, 2399])),
    ("crop183", ([17309, 9549, 5999], [17509, 9749, 6199])),
    ("crop416", ([8937, 13193, 13561], [9337, 13317, 13961])),
    ("crop417", ([4077, 14357, 8313], [4577, 14857, 8713])),
]

# Tuple format
transformed_data = [
    (name, (slice(x, xx), slice(y, yy), slice(z, zz)))
    for name, ([x, y, z], [xx, yy, zz]) in data
]



# ("crop_", ([], [])),
