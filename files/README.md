## Content
This folder contains the files required to train the model, and in the `/zattr` folder the .zattr files.
These files are taken from the OpenOrganelle dataset; only mithocondria files are downloaded. This can be found on the [janelia research group site](https://openorganelle.janelia.org/organelles/mito). 

### Download
The download is performed using the scripts found in scripts/mythocondria_download.py.
`get_all` gets all items from a list. It downloads files and converts them to HDF5. 

`get_some` only downloads the files with ground truth, or setting `inference = True`, also files with inference. What it does is:
- Use the list of files;
- Verify if the file has a ground truth (or inference); skip to next if it is missing;
- Read from zattr file the voxel size: download the compression that gets it closer to the `target_size`, defaults to 8. 
- Download the zarr file with a given compression;
- Transform the zarr into h5
Since the reading phase can take a long time, the list of all files found for download are stored in a txt file with the same name as the dataset, under `files/txt/{name}_to_download`.

### Statistics
`utils.py` contains the scripts required to fetch the information about the ground truth and the inference data (whether they are available or not) and about the voxel size. The `.zattrs` files are downloaded in the `fles/zattrs` folder; there is also generated the `data.json` file, which stores for each file:
`data.json` is the product of `read-zattrs`, which takes a list of `.zattrs` files downloaded by `get-zattrs` and collects the important metadata of the zarrs available for download.  

```
{
    name: foo,
    voxel_size: [z, y, x],
    has_groundtruth: True,
    has_inference: False
}
```
And stores the number of datasets with inference, ground truth and both:
```
{
    "inference": x, 
    "groundtruth": y, 
    "groundtruth_and_inference": z,
}
```