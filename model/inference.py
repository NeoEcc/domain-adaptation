import os
import torch
from torch_em.util import load_model
from inference_utils import check_inference, test_inference_loss
from UNet import model

# Keys for raw data and for labels
model_names = [
    "Source-AUNet-128-1",       # 0
    "AUNet-128-1-one-step-DA",  # 1
    "AUNet-128-1-two-steps-DA", # 2 
    "AUNet-128-1-finetuning-DA",# 3
    "AUNet-160-1-one-step-DA",  # 4
    "Source-AUNet-160-1",       # 5
    "Source-AUNet-256-1",       # 6
]
model_ID = 3
model_name = model_names[model_ID]
data_key = "raw_crop"
label_key = "label_crop/mito"
device = "cuda" if torch.cuda.is_available() else "cpu"
if model_ID <= 2:
    # Halo for 128x
    block_size = (90,)*3
    halo = (19,)*3
elif model_ID <= 5:
    # Halo for 160x
    block_size = (110,)*3
    halo = (25,)*3
elif model_ID > 5:
    # Halo for 256x
    block_size = (170,)*3
    halo = (43,)*3
else:
    raise RuntimeError(f"Unexpected model ID: {model_ID}")

# Path to the the folder with samples to test for inference
inference_path = "./files/test_crops/"

# Path to the folder where tostore the files modified for inference
save_inference_path = f"./files/test_inference/{model_name}/"

# Path to the checkpoints folder
save_path = "./model/"

# Path to the best version of the model
best_path = f"{save_path}checkpoints/{model_name}/best.pt"

# Load model
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

model = load_model(best_path, model, device = device)
model.eval()

if __name__ == "__main__":
    
    #   
    # Run inference
    #

    os.makedirs(save_inference_path, exist_ok=True)
    # Run inference on test files
    for file in os.listdir(inference_path):
        print("Checking inference for ", file, " with ", model_name)
        check_inference(
            model, f"{inference_path}{file}", f"{save_inference_path}{file}", 
            raw_key = data_key, label_key = label_key, postprocess = True
            )
    # Check test loss
    print("Inference completed. Calculating loss...")
    x = test_inference_loss(save_inference_path, label_key = label_key, average = True, memory_saving_level= 4)
    print(f"IoU: {x[0]}, dice: {x[1]}")

    old_test = [    # Files that used to be in the test set but were moved to training after the new test crop has been created
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_32.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_80.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_101.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_190.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_239.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_248.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_292.h5",
        "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/crop_355.h5",
    ]


