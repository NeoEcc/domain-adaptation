import os

from torch_em.util import load_model
from model_utils import check_inference
from UNet import model

model_name = "Anisotropic-3d-UNet-128-1"
data_key = "raw_crop"
block_size = (96,)*3
halo = (16,)*3


# Path to the samples to test for inference
inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_crops_copy/"

save_inference_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/test_inference/"

# Path to the checkpoints folder
save_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/model/"

# Path to the best version of the model
if model_name is None:
    from UNet import model_name
if data_key is None:
    from UNet import data_key

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

    for file, _ in zip(os.listdir(inference_path), range(10)):
        from UNet import loss_function
        print("Checking inference for ", file)
        loss = check_inference(model, f"{inference_path}{file}", f"{save_inference_path}{file}", test_function=loss_function)
