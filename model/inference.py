import os

from torch_em.util import load_model
from model_utils import check_inference, test_inference_loss
from UNet import model, device

# Keys for raw data and for labels
model_names = [
    "Source-AUNet-128-1",       # 0
    "AUNet-128-1-one-step-DA",  # 1
    "AUNet-128-1-two-steps-DA", # 2 
    "AUNet-160-1-one-step-DA",  # 3
]
model_ID = 0
model_name = model_names[model_ID]
data_key = "raw_crop"
label_key = "label_crop/mito"
if model_ID <= 2:
    # Halo for 128x
    block_size = (90,)*3
    halo = (19,)*3
else:
    # Halo for 160x
    block_size = (110,)*3
    halo = (25,)*3

# Path to the the folder with samples to test for inference
inference_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_crops/"

# Path to the folder where tostore the files modified for inference
save_inference_path = f"/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/test_inference/{model_name}/"

# Path to the checkpoints folder
save_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/model/"

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
    # for file in ["crop_32.h5", "crop_80.h5"]:
        print("Checking inference for ", file, " with ", model_name)
        check_inference(
            model, f"{inference_path}{file}", f"{save_inference_path}{file}", 
            raw_key = data_key, label_key = label_key, postprocess = True
            )
    # Check test loss
    x = test_inference_loss(save_inference_path, label_key = label_key, average = True)
    print(f"IoU: {x[0]}, dice: {x[1]}")
