## Content
This folder will contain the files required for training and other miscellaneous files.

Several scripts will produce files in subfolders here, such as in `/crops` (produced by defauly by `scripts/resize_to_target`), as well as the useful information produced by several utility functions in `/txt`.

It can also contain the files downloaded from the the OpenOrganelle dataset; using `scripts/get_filtered_from_bucket` or `scripts/get_all`, only mithocondria files or all files are downloaded. 

The whole dataset can be found on the [janelia research group site](https://openorganelle.janelia.org/organelles/mito). 

### Download
The download is performed using the scripts found in scripts/mythocondria_download.py.

`get_all` gets all items from a list. It downloads files and converts them to HDF5. 

`get_some` only downloads the files with ground truth, or setting `inference = True`, also files with inference. For all files in the list, it checks if there is ground truth or inference; if yes, determines the right voxel size and downloads the file in the specified folder, then converts it to HDF5.
Since the reading phase can take a long time, the list of all files found for download are stored in a txt file with the same name as the dataset, under `files/txt/{name}_to_download`. The path to this file can be passed as a parameter to the function to skip the reading process. 

### Statistics
`utils.py` also contains functions to read dataset information relevant for training, such as resolution and voxel size.

`read_attributes_h5` reads the attributes of all HDF5 files in a given path, and prints the relevant statistics in a json file specified by the user as:
```
    name: foo,
    resolution: [z, y, x]
    voxel_size: [z, y, x],
    translation: [z, y, x],
```
