import torch_em
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import RAdam
from torch_em.model import AnisotropicUNet
from torch_em.trainer import DefaultTrainer
from torch_em.util import load_model
from model_utils import *

#
# Hyperparameters
#

model_name = "Anisotropic-3d-UNet-128-1"
# name-patch_shape-batch_size

learning_rate = 1.0e-4      # learning rate for the optimizer
batch_size = 1              # batch size for the dataloader
epochs = 5000               # number of epochs to train the model for
random_seed = 42            # random seed for reproducibility
classes = ["mito"]          # list of classes to segment
patch_shape = (128,)*3      # Patch shape - modified in various runs
val_split = 0.1             # Fraction of the data to use for validation
num_workers = 2             # Limit number of cpus
loss_function = torch_em.loss.DiceLoss()
metric_function = torch_em.loss.DiceLoss()
device = "cuda"             # Device required for training

#
# Paths
#

# Path of the training folder
data_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/mito_crops/"

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the samples to test for inference
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/inference_crops/"

# Path to the best version of the model
# Give none to train from scratch
best_path = None
best_path = f"{save_path}checkpoints/{model_name}/best.pt"

# Keys for raw data and for labels
data_key = "raw_crop"
label_key = "label_crop/mito"


#
# Model
# 

in_channels = 1             # 3D b&w
out_channels = 2            # One for foreground, one for boundaries
scale_factors = [           # All the same, since on average all dimensions have the same nm/voxel
    [2, 2, 2], 
    [2, 2, 2], 
    [2, 2, 2], 
    [2, 2, 2]
    ]

model = AnisotropicUNet(
    in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors, final_activation="Sigmoid"
)

if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)
# Load weights if any
if best_path is not None:
    model = load_model(best_path, model)


#
# Data loaders and trainer
#

# OR adamw
optimizer = RAdam(
            model.parameters(), lr=learning_rate, decoupled_weight_decay=True
        )

train_loader, val_loader = get_dataloader(
    directory_to_path_list(data_path), 
    data_key,
    label_key, 
    val_split, 
    patch_shape, 
    batch_size,
    num_workers = num_workers
    )

scheduler = ReduceLROnPlateau(optimizer)

trainer = DefaultTrainer(
    name = model_name,
    train_loader = train_loader,
    val_loader = val_loader,
    model = model,
    loss = loss_function,
    optimizer = optimizer,
    early_stopping= 25,
    lr_scheduler= scheduler,
    metric = metric_function,
    device = device,
    save_root = save_path
)

if __name__ == "__main__":

    #
    # Train
    #

    print(f"Training {model_name} for {epochs} epochs with a lr of {learning_rate}, batch size of {batch_size} and patch shape of {patch_shape}")

    trainer.fit(
        epochs=epochs,
        save_every_kth_epoch=20,
    )
