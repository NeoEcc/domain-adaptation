import os

from semisupervised_utils import semisupervised_training
from model_utils import directory_to_path_list
from sklearn.model_selection import train_test_split

from UNet import model

# Domain adaptation modes - comment or uncomment to select
domain_adaptation_mode = "one-step"
    # One step uses the source domain in the labeled data and target domain in the unlabeled data to train a model from scratch.
    # Performs pure semi-supervised training
# domain_adaptation_mode = "two-steps"
    # Two steps uses the source domain labeled data to train a model, 
    # and unlabeled and few labeled data from the target domain to refine the same model. 

# Name of the model to be used as a base (for two-steps) and the new model and path to checkpoint
old_model_name = "Source-AUNet-128-1"
best_path = None
# new_model_name = f"{old_model_name}-{domain_adaptation_mode}-DA"
new_model_name = "AUNet-128-1-one-step-DA"

#
# Hyperparameters
#
patch_shape = (128,)*3
lr = 1.0e-4
val_split = 0.15
batch_size = 1

# Path to the checkpoints folder
save_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/model/"
# Internal paths
raw_key = "/raw_crop"
label_key = "/label_crop/mito"
# Paths to the data in the target domain
unlabeled_folder_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/target_unlabeled"
labeled_folder_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/target_labeled"
# Path to all labeled crops
source_labeled_folder_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/source_labeled"

if domain_adaptation_mode == "two-steps":
    # Get the data paths
    unlabeled_data_paths = directory_to_path_list(unlabeled_folder_path)
    labeled_data_paths = directory_to_path_list(labeled_folder_path)
    unlabeled_train_paths, unlabeled_val_paths = train_test_split(unlabeled_data_paths, test_size = val_split, random_state = 42)
    labeled_train_paths, labeled_val_paths = train_test_split(labeled_data_paths, test_size = val_split, random_state = 42)

    # Path to the best version of the previous model
    best_path = f"{save_path}checkpoints/{old_model_name}/best.pt"

    # Load model for training
    if best_path is None or not os.path.exists(best_path):
        raise ValueError("Path to model must be given and cannot be empty: " + best_path)
    # Feature moved into the training function; keeping this check 

elif domain_adaptation_mode == "one-step":
    # Get the data paths
    unlabeled_data_paths = directory_to_path_list(unlabeled_folder_path)
    
    labeled_data_paths = directory_to_path_list(source_labeled_folder_path) + directory_to_path_list(labeled_folder_path) 
    unlabeled_train_paths, unlabeled_val_paths = train_test_split(unlabeled_data_paths, test_size = val_split, random_state = 42)
    labeled_train_paths, labeled_val_paths = train_test_split(labeled_data_paths, test_size = val_split, random_state = 42)
else:
    raise ValueError(f"Expected 'one-step' or 'two-steps', got {domain_adaptation_mode}")
    

if __name__ == "__main__":
    model.train()
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
        batch_size = batch_size,
        lr = lr
    )