import numpy as np
import torch.nn as nn
import torch_em
import os
import torch
from torch.optim import RAdam
from torch_em.model import AnisotropicUNet, get_vision_transformer
from torch_em.trainer import DefaultTrainer
from torch_em.util import load_model
from sklearn.model_selection import train_test_split
from model_utils import *
import multiprocessing

# Switch between inference and training

is_inference = True

#
# Hyperparameters
#

model_name = "Anisotropic-3d-UNet-fresh"

learning_rate = 1.0e-4      # learning rate for the optimizer
batch_size = 1              # batch size for the dataloader
epochs = 1#5000                # number of epochs to train the model for
iterations_per_epoch = 100 # number of iterations per epoch
random_seed = 42            # random seed for reproducibility
classes = ["mito"]          # list of classes to segment
patch_shape = (128,)*3      # !! To be studied !!
val_split = 0.1
num_workers = 2             # Limit number of cpus
loss_function = torch_em.loss.DiceLoss()
metric_function = torch_em.loss.DiceLoss()
device = "cuda"             # Device required for training

#
# Paths
#

# Path of the training folder
data_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops"
# data_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops"

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the samples to test for inference
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/inference_crops/"

# Path to the best version of the model
best_path = f"{save_path}checkpoints/{model_name}/latest.pt"

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

best_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/checkpoints/Anisotropic-3d-UNet-fresh/"

# Load weights if any
if best_path is not None and os.path.exists(best_path):
    model = load_model(best_path, model)

#
# Data loaders and trainer
#

# OR adamw
optimizer = RAdam(
            model.parameters(), lr=learning_rate, decoupled_weight_decay=True
        )

train_loader, val_loader = get_dataloader(directory_to_path_list(data_path), data_key, label_key, patch_shape, val_split, batch_size, num_workers = num_workers)

trainer = DefaultTrainer(
    name = model_name,
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    optimizer = optimizer,
    metric = metric_function,
    loss = loss_function,
    device = device,
    save_root = save_path
)
if __name__ == "__main__":
    if not is_inference:
        #
        # Train
        #

        trainer.fit(
            epochs=epochs,
            save_every_kth_epoch=5,
        )

    else:
        #   
        # Test inference
        #

        for file in os.listdir(inference_path):
            check_inference(model, f"{inference_path}{file}")

    
# Overfitting curve:
# Epoch 35:  average [s/it]: 0.118461, current metric: 1.999343, best metric: 1.979291:   0%|
# Epoch 61:  average [s/it]: 0.271925, current metric: 2.000000, best metric: 1.979291:   1%|▉   
# Epoch 71:  average [s/it]: 0.120462, current metric: 2.000000, best metric: 1.979289:   1%|█   
# Epoch 107: average [s/it]: 0.118276, current metric: 2.000000, best metric: 1.917568:   2%|█▌
# Epoch 128: average [s/it]: 0.118628, current metric: 2.000000, best metric: 1.885475:   3%|█▊
# Epoch 145: average [s/it]: 0.309648, current metric: 1.924561, best metric: 1.885475:   3%|██     
# Epoch 154: average [s/it]: 0.120972, current metric: 2.000000, best metric: 1.885475:   3%|██▏
# Epoch 164: average [s/it]: 0.121674, current metric: 2.000000, best metric: 1.885475:   3%|██
# Epoch 204: average [s/it]: 0.126037, current metric: 1.998157, best metric: 1.885475:   4%|██▉
# Epoch 246: average [s/it]: 0.119816, current metric: 2.000000, best metric: 1.885475:   5%|███▌ 
# Epoch 282: average [s/it]: 0.118742, current metric: 2.000000, best metric: 1.885475:   6%|████
# Epoch 307: average [s/it]: 0.120881, current metric: 2.000000, best metric: 1.885475:   6%|████▍
# Epoch 332: average [s/it]: 0.122422, current metric: 2.000000, best metric: 1.885475:   7%|████▋
# Epoch 391: average [s/it]: 0.121376, current metric: 2.000000, best metric: 1.885475:   8%|█████▌
# Epoch 460: average [s/it]: 0.126515, current metric: 1.988828, best metric: 1.785747:   9%|██████▌
# Epoch 550: average [s/it]: 0.122639, current metric: 1.922272, best metric: 1.716292:  11%|███████▊ 
# Epoch 736: average [s/it]: 0.122496, current metric: 1.964961, best metric: 1.716292:  15%|██████████