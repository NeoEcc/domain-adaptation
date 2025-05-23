# TODO: move inference here
import os
from torch_em.util import load_model
from model_utils import check_inference
from UNet import model

model_name = "Anisotropic-3d-UNet-256"

# Path to the samples to test for inference
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/inference_crops/"

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the best version of the model
if model_name is None:
    from UNet import model_name
best_path = f"{save_path}checkpoints/{model_name}/best.pt"


# Load model
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

model = load_model(best_path, model)

if __name__ == "__main__":
    #   
    # Run inference
    #

    # for file in os.listdir(inference_path):
    for file in ["crop_145.h5"]: # "crop_368.h5", "crop_258.h5", 
        check_inference(model, f"{inference_path}{file}")
