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


# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the data in the target domain
target_data_path = "/scratch-grete/projects/nim00007/data/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr"

if old_model_name is None:
    from UNet import model_name as old_model_name
if save_path is None:
    from UNet import save_path

# Path to the best version of the previous model
best_path = f"{save_path}checkpoints/{old_model_name}/best.pt"

# Load model for training
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

model = load_model(best_path, model)
model.train()

if __name__ == "__main__":
    semisupervised_training(
        name = old_model_name, model = model, train_paths = (target_data_path), val_paths = (target_data_path), 
    )

# - ROI for mitochondria example:
# 	- x [30000:80000]
# 	- y [20000:50000]
# 	- z [35000:50000]
