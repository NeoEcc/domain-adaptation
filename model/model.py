import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data
import os
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer
from sklearn.model_selection import train_test_split

#
# CODE FROM 3D-UNet-Training.ipynb
# https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb
#

# Prepare data

all_paths_raw = os.listdir( "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/")
print (all_paths_raw[0])
all_paths = []
for i in range(len(all_paths_raw)):
    all_paths.append(f"/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/crops/{all_paths_raw[i]}")
print (all_paths[0])

train_data_paths, val_data_paths = train_test_split(all_paths, test_size=0.2, random_state=42)

print("all_paths", all_paths.__len__())
print("train_data_paths", train_data_paths.__len__())
print("val_data_paths", val_data_paths.__len__())

data_key = "raw_crop"
train_label_paths = train_data_paths
val_label_paths = val_data_paths
label_key = "label_crop/mito"
patch_shape = (32, 256, 256) # To be modified?

train_rois = None
val_rois = None

# Data check
def check_data(data_paths, label_paths, rois):
    # print("Loading the raw data from:", data_paths, data_key)
    # print("Loading the labels from:", label_paths, label_key)
    try:
        torch_em.default_segmentation_dataset(data_paths, data_key, label_paths, label_key, patch_shape, rois=rois)
    except Exception as e:
        print("Loading the dataset failed with:")
        raise e

check_data(train_data_paths, train_label_paths, train_rois)
check_data(val_data_paths, val_label_paths, val_rois)

assert len(patch_shape) == 3

# What to do
# Whether to add a foreground channel (1 for all labels that are not zero) to the target.
foreground = True
# Whether to add affinity channels (= directed boundaries) or a boundary channel to the target.
# Note that you can choose at most of these two options.
affinities = False
boundaries = True

offsets = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-2, 0, 0], [0, -3, 0], [0, 0, -3],
    [-3, 0, 0], [0, -9, 0], [0, 0, -9]
]

label_transform, label_transform2 = None, None
if affinities:
    label_transform2 = torch_em.transform.label.AffinityTransform(
        offsets=offsets, add_binary_target=foreground, add_mask=True
    )
elif boundaries:
    label_transform = torch_em.transform.label.BoundaryTransform(
        add_binary_target=foreground
    )
elif foreground:
    label_transform = torch_em.transform.label.labels_to_binary

batch_size = 1
loss = "dice"
metric = "dice"

def get_loss(loss_name):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss()
        elif loss == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name
    
    # we need to add a loss wrapper for affinities
    if affinities:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=torch_em.loss.ApplyAndRemoveMask()
        )
    return loss_function


loss_function = get_loss(loss)
metric_function = get_loss(metric)

kwargs = dict(
    ndim=3, patch_shape=patch_shape, batch_size=batch_size,
    label_transform=label_transform, label_transform2=label_transform2
)

train_loader = torch_em.default_segmentation_loader(
    train_data_paths, data_key, train_label_paths, label_key,
    rois=train_rois, **kwargs
)
val_loader = torch_em.default_segmentation_loader(
    val_data_paths, data_key, val_label_paths, label_key,
    rois=val_rois, **kwargs
)

assert train_loader is not None, "Something went wrong"
assert val_loader is not None, "Something went wrong"

# UNet preparation

scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]

initial_features = 32
final_activation = None

# If you leave the in/out_channels as None an attempt will be made to automatically deduce these numbers. 
in_channels = None
out_channels = None

if final_activation is None and loss == "dice":
    final_activation = "Sigmoid"
    print("Adding a sigmoid activation because we are using dice loss")

if in_channels is None:
    in_channels = 1

if out_channels is None:
    if affinities:
        n_off = len(offsets)
        out_channels = n_off + 1 if foreground else n_off
    elif boundaries:
        out_channels = 2 if foreground else 1
    elif foreground:
        out_channels = 1
    assert out_channels is not None, "The number of out channels could not be deduced automatically. Please set it manually in the cell above."

print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
model = AnisotropicUNet(
    in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors, final_activation=final_activation
)

# Configure training
experiment_name = "Test-1"
n_iterations = 3
learning_rate = 1.0e-4

trainer = torch_em.default_segmentation_trainer(
    name=experiment_name, model=model,
    train_loader=train_loader, val_loader=val_loader,
    loss=loss_function, metric=metric_function,
    learning_rate=learning_rate,
    mixed_precision=True,
    log_image_interval=50,
    # logger=None
)
print(type(experiment_name))

print("OK")
# Debugging: Print the types of variables that could be None
print("Type of trainer:", type(trainer))
print("Type of train_loader:", type(train_loader))
print("Type of val_loader:", type(val_loader))
print("Type of model:", type(model))
print("Type of loss_function:", type(loss_function))
print("Type of metric_function:", type(metric_function))

trainer.fit(n_iterations)

# The folder where the bioimageio model will be saved (as a .zip file).
# If you run in google colab you should adapt this path to your google drive so that you can download the saved model.
export_folder = "./test-model"

# Whether to convert the model weights to additional formats.
# Currently, torchscript and onnx are support it and this will enable running the model
# in more software tools.
additional_weight_formats = None
# additional_weight_formats = ["torchscript"]

doc = None
# write some markdown documentation like this, otherwise a default documentation text will be used
# doc = """#My Fancy Model
# This is a fancy model to segment shiny objects in images.
# """

import torch_em.util.modelzoo

for_dij = additional_weight_formats is not None and "torchscript" in additional_weight_formats

training_data = None

pred_str = ""
if affinities:
    pred_str = "affinities and foreground probabilities" if foreground else "affinities"
elif boundaries:
    pred_str = "boundary and foreground probabilities" if foreground else "boundaries"
elif foreground:
    pred_str = "foreground"

default_doc = f"""#{experiment_name}

This model was trained with [the torch_em 3d UNet notebook](https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).
"""
if pred_str:
    default_doc += f"It predicts {pred_str}.\n"

training_summary = torch_em.util.get_training_summary(trainer, to_md=True, lr=learning_rate)
default_doc += f"""## Training Schedule

{training_summary}
"""

if doc is None:
    doc = default_doc

torch_em.util.modelzoo.export_bioimageio_model(
    trainer, export_folder, input_optional_parameters=True,
    for_deepimagej=for_dij, training_data=training_data, documentation=doc
)
torch_em.util.modelzoo.add_weight_formats(export_folder, additional_weight_formats)