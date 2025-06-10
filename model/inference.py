import os

from torch_em.util import load_model
from model_utils import check_inference, test_inference_loss
from UNet import model

model_name = "Anisotropic-3d-UNet-160-1"
# Keys for raw data and for labels
data_key = "raw_crop"
label_key = "label_crop/mito"
block_size = (120,)*3
halo = (20,)*3

# Path to the the folder with samples to test for inference
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops/"

# Path to the folder where tostore the files modified for inference
save_inference_path = f"/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_inference/{model_name}/"

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

if model_name is None:
    from UNet import model_name
if data_key is None:
    from UNet import data_key

# Path to the best version of the model
best_path = f"{save_path}checkpoints/{model_name}/best.pt"

# Load model
if best_path is not None and not os.path.exists(best_path):
    raise ValueError("Path to model is empty: " + best_path)

model = load_model(best_path, model)
model.eval()

if __name__ == "__main__":
    
    #   
    # Run inference
    #

    # Run inference on test files
    for file in os.listdir(inference_path):
        print("Checking inference for ", file, " with ", model_name)
        check_inference(
            model, f"{inference_path}{file}", f"{save_inference_path}{file}", 
            raw_key = data_key, label_key = label_key
            )
    # Check test loss
    x = test_inference_loss(save_inference_path, label_key = label_key, average = True)
    print(f"IoU: {x[0]}, dice: {x[1]}")
