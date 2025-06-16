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
from semisupervised_utils import semisupervised_training

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

    # Coordinates and names of samples moved to files/txt/labeled_data...
