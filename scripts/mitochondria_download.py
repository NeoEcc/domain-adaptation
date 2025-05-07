from utils import get_filtered_from_bucket

if __name__ == "__main__":
    # Hardcoded list of all datasets available with mitochondria
    names = [
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
        "jrc_cos7-11",            # 90.6 GB
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
        "jrc_hela-2",             # 70 GB  
        "jrc_macrophage-2",       # 96 GB   
        "jrc_jurkat-1",           # 123 GB  
        "jrc_hela-3",             # 133 GB  
        "jrc_ctl-id8-1",          # 235 GB  
        "jrc_mus-kidney"          # unknown 
        "jrc_mus-liver",          # 1.12 TB 
        "jrc_sum159-1",           # 13.9 TB
    ]

    bucket_str = "s3://janelia-cosem-datasets"
    folder_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/"

    get_filtered_from_bucket(names_with_labels, folder_path, bucket_str, max_threads=64)