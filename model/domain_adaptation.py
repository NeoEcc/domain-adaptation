# This refers to the first method studied: first train the model and then run the semisupervised training on top of that

# TODO: Create new file for the second method

# TODO:
#  Import old model
#  Import parameters: patch size, 
#  Define new parameters?
#  Define new paths
#  Create new dataloader
#    Find out the correct number of samples, probably 10x the number of pixel (avg 165x) or 10x the number of samples
#    134 * 10 * 165^3 / 128^3 = 2870 samples
#    Finalized to 80 512x crops, which are 80x64 = 5120 120x crops. 
#  Train 
import os

from torch_em.util import load_model
from semisupervised_utils import semisupervised_training
from model_utils import directory_to_path_list
from sklearn.model_selection import train_test_split

from UNet import model

# Name of the model to be used as a base and the new model
old_model_name = "Anisotropic-3d-UNet-128-1"
new_model_name = f"{old_model_name}-DA"

# Patch shape of the model
patch_shape = (128,)*3

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Learning rate
lr = 1.0e-4

# Paths to the data in the target domain
val_split = 0.15
unlabeled_folder_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/labeled_crops"
labeled_folder_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/unlabeled_crops"
raw_key = "/raw_crop"
label_key = "/label_crop/mito"
unlabeled_data_paths = directory_to_path_list(unlabeled_folder_path)
labeled_data_paths = directory_to_path_list(labeled_folder_path)
unlabeled_train_paths, unlabeled_val_paths = train_test_split(unlabeled_data_paths, val_split, random_state = 42)
labeled_train_paths, labeled_val_paths = train_test_split(labeled_data_paths, val_split, random_state = 42)

# Path to the best version of the previous model
best_path = f"{save_path}checkpoints/{old_model_name}/best.pt"


# Load model for training
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

# Feature moved into the training function; keeping this check 

# model = load_model(best_path, model)
# model.train()

if __name__ == "__main__":
    semisupervised_training(
        name = old_model_name, 
        model = model, 
        train_paths = (labeled_train_paths, unlabeled_train_paths), 
        val_paths = (labeled_val_paths, unlabeled_val_paths), 
        label_key = label_key, 
        patch_shape = patch_shape, 
        save_root = save_path,
        raw_key = raw_key,
        load_path = best_path,
        batch_size = 1,
        lr = lr
    )