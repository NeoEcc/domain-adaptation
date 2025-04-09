import quilt3 as q3
import os
import json

from zarr_utils import zarr_to_h5
from utils import get_best_size, get_folder_parallel


# # # Functions definition

def get_some(names, path, bucket_str, inference = False, target_size = 8, max_threads = None):
    """
    Steps:
    - Iterate over the list of names
    - Check whether there exists a groundtruth folder
    - Read the .zargs file, get the voxel size
    - Decide the downscale level to get closer to the target size
    - Download the correct downscaled version of the dataset
    - Convert to h5

    Parameters:
    names (list): List of names to download
    path (str): Path to the folder where the data will be downloaded. ENDS WITH "/"
    bucket_str (str): Bucket name
    inference (bool): Whether to download datasets that have no groundtruth but have inference
    target_size (int): Target size in nm between voxels
    """
    b = q3.Bucket(bucket_str)
    # Iterate
    for name in names: 
        print("[get_some]: Handling " + name)
        try:
            # Check if there is a groundtruth 
            # If not, check for inference if required
            # Otherwise, continue to next file
            try:
                data = b.ls(f"{name}/{name}.zarr/recon-1/labels/groundtruth/")
            except:
                print("groundtruth not found for " + name)
                if inference: 
                    try:
                        data = b.ls(f"{name}/{name}.zarr/recon-1/labels/inference/")
                    except:
                        print("inference not found for " + name)
                        continue
                else:  
                    continue
            # If here, dataset was deemed downloadable. 
            # Get the attributes file: read the voxel size and decide the scale level. 
            if not os.path.isfile(f"{path}/zattrs/{name}.zattrs"):
                    b.fetch(f"{name}/{name}.zarr/recon-1/em/fibsem-uint8/.zattrs", f"{path}/zattrs/{name}.zattrs")
            try:
                with open(f"{path}/zattrs/{name}.zattrs") as f:
                    data = json.load(f)
                name = os.path.splitext(name + ".zarr")[0]
                voxel_size = data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1]
            except:
                print("Failed to fetch file")
            # Try to get as close as possible to target_size
            
            compression = get_best_size(voxel_size, target_size)
            download_string = f"s{compression}"
            filename = f"{path}{name}.zarr" # !!! with / final cannot be used
            folders_to_ignore = ["masks"]
            if not inference:
                folders_to_ignore.append("inference")
            get_folder_parallel(b, 
                                f"{name}/{name}.zarr/", 
                                filename, 
                                download_string, max_threads, folders_to_ignore)
            # Convert to h5
            zarr_to_h5(filename) 
            print("Converted" + name)
        except Exception as e:
            print(f"Failed to download {name}: {str(e)}.")

def get_all(names, bucket, folder):
    b = q3.Bucket(bucket)
    n_downloaded = 0
    for name in names:
        downloaded = False
        try:
            # Download
            filename = folder + name + ".zarr" # !!! / is an important difference, not the same as destination_name
            folders_to_ignore = ["masks"]
            get_folder_parallel(b, 
                                f"{name}/{name}.zarr/recon-1/", 
                                filename, 
                                folder, 128, folders_to_ignore)
            downloaded = True
            # Convert
            h5_name = folder + name + ".h5"
            zarr_to_h5(filename, h5_name)
            n_downloaded += 1
        except Exception as e:
            if downloaded: 
                print(f"Failed to download {name}: {str(e)}.")
            else:
                print(f"Failed to convert {name}: {str(e)}.")
    print(f"Downloaded {n_downloaded} files")

if __name__ == "__main__":
    print("[mitochondria_download.py]")
    names = [
        # "jrc_hela-1",             # 31 GB
        # "jrc_hela-2",             # 70 GB     # Smallest with groundtruth
        # "jrc_hela-21",            # 9.8 GB
        # "jrc_hela-22",            # 16 GB
        # "jrc_hela-3",             # 133 GB
        
        # "jrc_hela-bfa",           # 14 GB
        # "jrc_hela-h89-1",         # 44 GB
        # "jrc_hela-h89-2",         # 57 GB
        "jrc_mus-kidney",
        "jrc_mus-pancreas-1",
        "jrc_mus-pancreas-2",
        "jrc_mus-pancreas-3",
        "jrc_mus-sc-zp104a",    # 340 GB  
        "jrc_mus-sc-zp105a",    # 258 GB
        "jrc_ccl81-covid-1",    # 356 GB
        "jrc_choroid-plexus-2", # 507 GB
        "jrc_cos7-11",          # 90.6 GB
        "jrc_ctl-id8-1",        # 235 GB
        "jrc_ctl-id8-2",        # 28.2 GB
        "jrc_ctl-id8-3",        # 39.2 GB
        "jrc_ctl-id8-4",        # 32.7 GB
        "jrc_ctl-id8-5",        # 18.3GB # Forgot comma, fused together and failed to fetch 
        "jrc_dauer-larva",      # 315 GB
        "jrc_fly-larva-1",      # 9.72 TB
        "jrc_fly-acc-calyx-1",  # 4.81 TB
        "jrc_fly-fsb-1",        # 2.45 TB
        "jrc_fly-mb-z0419-20",  # 2.45 TB
        "jrc_sum159-1",         # 13.9 TB
        "jrc_hela-4",           # 318 GB
        "jrc_jurkat-1",         # 123 GB
        "jrc_macrophage-2",     # 96 GB     
        "jrc_mus-liver",        # 1.12TB    
    ]

    sorted_names = [
        "jrc_hela-h89-1",
        "jrc_hela-h89-2",
        "jrc_mus-kidney",
        "jrc_mus-pancreas-1",
        "jrc_mus-pancreas-2",
        "jrc_mus-pancreas-3",
        "jrc_hela-21",            # 9.8 GB
        "jrc_fly-larva-1",        # 9.72 TB
        "jrc_hela-22",            # 16 GB
        "jrc_ctl-id8-5",          # 18.3 GB
        "jrc_ctl-id8-2",          # 28.2 GB
        "jrc_hela-1",             # 31 GB
        "jrc_ctl-id8-4",          # 32.7 GB
        "jrc_ctl-id8-3",          # 39.2 GB
        "jrc_hela-bfa",           # 14 GB
        "jrc_hela-2",             # 70 GB
        "jrc_macrophage-2",       # 96 GB
        "jrc_cos7-11",           # 90.6 GB
        "jrc_jurkat-1",           # 123 GB
        "jrc_hela-3",             # 133 GB
        "jrc_ctl-id8-1",          # 235 GB
        "jrc_mus-sc-zp105a",      # 258 GB
        "jrc_mus-sc-zp104a",      # 340 GB
        "jrc_hela-4",             # 318 GB
        "jrc_dauer-larva",        # 315 GB
        "jrc_ccl81-covid-1",      # 356 GB
        "jrc_choroid-plexus-2",   # 507 GB
        "jrc_mus-liver",          # 1.12 TB
        "jrc_fly-acc-calyx-1",    # 4.81 TB
        "jrc_fly-fsb-1",          # 2.45 TB
        "jrc_fly-mb-z0419-20",    # 2.45 TB
        "jrc_sum159-1",           # 13.9 TB
    ]

    names_with_labels = [
        "jrc_hela-2",             # 70 GB   # 12GB after only 8nm # 36GB in h5??
        "jrc_macrophage-2",       # 96 GB   # 15
        "jrc_jurkat-1",           # 123 GB  # 20
        "jrc_hela-3",             # 133 GB  # 18
        "jrc_ctl-id8-1",          # 235 GB  
        "jrc_mus-kidney"          # unknown # 
        # "jrc_mus-liver",          # 1.12 TB # Too big
        # "jrc_sum159-1",           # 13.9 TB
    ]

    bucket_str = "s3://janelia-cosem-datasets"
    folder_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/"

    get_some(names_with_labels, folder_path, bucket_str, max_threads=64)