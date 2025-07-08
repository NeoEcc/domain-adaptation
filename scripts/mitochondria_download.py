import quilt3 as q3

if __name__ == "__main__":
    
    bucket_str = "s3://janelia-cosem-datasets"
    folder_path = "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/"

    b = q3.Bucket("s3://janelia-cosem-datasets")
    b.fetch("jrc_ctl-id8-2/jrc_ctl-id8-2.zarr/", "/mnt/lustre-emmy-ssd/projects/nim00007/data/mitochondria/files/example_dataset/jrc_ctl-id8-2.zarr/")
