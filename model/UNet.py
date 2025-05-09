import numpy as np
import torch.nn as nn
import torch_em
import os
import torch
from torch.optim import RAdam
from torch_em.model import AnisotropicUNet, get_vision_transformer
from torch_em.trainer import DefaultTrainer
from sklearn.model_selection import train_test_split
from model_utils import *

#
# Paths
#

# data_path = "../../mitochondria/files/test_crops/"
data_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops"
# save_path = "../model/checkpoints/"
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/inference_crops/"
data_key = "raw_crop"
label_key = "label_crop/mito"


#
# Hyperparameters
#

learning_rate = 1.0e-4      # learning rate for the optimizer
batch_size = 8              # batch size for the dataloader
epochs = 5                # number of epochs to train the model for
iterations_per_epoch = 2 # number of iterations per epoch
random_seed = 42            # random seed for reproducibility
classes = ["mito"]          # list of classes to segment
patch_shape = (64,)*3      # !! To be studied !!
val_split = 0.2
num_workers = 8
loss_function = torch_em.loss.DiceLoss()
metric_function = torch_em.loss.DiceLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Anisotropic-3d-UNet"

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

# trainer = torch_em.default_segmentation_trainer( # Actually calls an instance of DefaultTrainer
#     name = model_name, 
#     model = model,
#     train_loader = train_loader, 
#     val_loader = val_loader,
#     loss = loss_function, 
#     metric = metric_function,
#     learning_rate = learning_rate,
#     mixed_precision = True,
#     log_image_interval = 50,
#     device = device
# )

#
# Train
#

trainer.fit(
    epochs=epochs,
    save_every_kth_epoch=1,
)

#
# Test inference
#

for file in os.listdir(inference_path):
    check_inference(model, f"{inference_path}{file}")