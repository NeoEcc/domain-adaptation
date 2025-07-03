# First functions adapted from synapse-net to work with zarrs
# https://github.com/computational-cell-analytics/synapse-net/blob/main/synapse_net/training/semisupervised_training.py

from typing import Optional, Tuple
from torchvision import transforms
from torch_em.util import load_model


import os
import torch
import torch_em
import torch_em.self_training as self_training

def get_sub_rois(original_roi: Tuple[slice], crop_size: Tuple[int]):
    """
    From a large ROI, extracts as many smaller ROI as possible in an array.

    Args:
        original_roi (Tuple(slice)): tuple of slices representing the ROI from which to get the crops
        crop_size (Tuple(int)): size of the sub-crops
    """   
    # Input checks
    ndim = len(crop_size)
    if ndim != 2 and ndim != 3:
        raise ValueError("Only 2d or 3d ROIs are supported")
    if ndim != len(original_roi):
        raise ValueError("Dimensions of ROI must be the same as the crop")
    roi_shape = tuple(s.stop - s.start for s in original_roi)
    if any(s < c for s, c in zip(roi_shape, crop_size)): 
        raise ValueError(f"Inserted ROI is smaller than the crop size ({roi_shape},{crop_size})")
    elif all(s == c for s, c in zip(roi_shape, crop_size)):
        print("Warining: size of original crop and crop size are the same")
        return [original_roi] 
    elif any(s % c != 0 for s, c in zip(roi_shape, crop_size)):
        print("Warining: part of the crop will be discarded. ", crop_size, " is not a multiple of ", roi_shape)

    slices = []

    # Per-dimension division, can iterate indefinitely since the checks have been made
    x = 0
    while x <= original_roi[0].stop - crop_size[0]:
        y = 0
        while y <= original_roi[1].stop - crop_size[1]:
            temp_slice = (
                slice(x, x + crop_size[0]),
                slice(y, y + crop_size[1])
            )
            if ndim == 3:
                z = 0
                while z <= original_roi[2].stop - crop_size[2]:
                    temp_slice_z = temp_slice + (slice(z, z + crop_size[2]),)
                    slices.append(temp_slice_z)
                    z += crop_size[2]
            else:
                slices.append(temp_slice)
            y += crop_size[1]
        x += crop_size[0]
    # Return
    return slices

def get_unsupervised_loader(
    data_paths: list[str],
    raw_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples_epoch: Optional[int],
    # roi = None,
    # blacklist_roi = None,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_path: The filepaths to the hdf5 or zarr files with the training data.
        raw_key: The key that holds the raw data inside of the hdf5 or zarr.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        n_samples_epoch: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.

        Not implemented in this version:
        # roi: specify a region of interest, can be None.
        # blacklist_roi: list of regions to be avoided, as array of tuples of slices, or None

    Returns:
        The PyTorch dataloader.
    """

    ndim = 3

    # Transforms and augmentations definition
    # TODO: implement strong augmentations
    raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=ndim)
    augmentations = (weak_augmentations(), weak_augmentations())
    
    # HDF5 version
    # Each sample is 512x. Must extract 64 128x crops from each.
    crops_shape = (slice(0,512),)*3
    target_shape = (128,)*3
    rois = get_sub_rois(crops_shape, target_shape)
    
    # Calculate samples per dataset
    total_datasets = len(data_paths) * len(rois)
    if n_samples_epoch is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples_epoch / total_datasets)
    
    datasets = []
    for current_path in data_paths:
        for roi in rois:
            datasets.append(
                torch_em.data.RawDataset(current_path, raw_key, patch_shape, raw_transform, transform,
                                        augmentations=augmentations, roi=roi,
                                        ndim=ndim, n_samples=n_samples_per_ds)
            )

    # ### MODIFIED HERE TO ADAPT TO ZARR  
    # datasets = [
    #     torch_em.data.RawDataset(data_path, raw_key, patch_shape, raw_transform, transform,
    #                              augmentations=augmentations, roi=get_random_roi(roi, patch_shape, blacklist_roi),
    #                              ndim=ndim, n_samples=n_samples_per_ds)
    #     for _ in range(n_samples_epoch)
    # ]
    ds = torch.utils.data.ConcatDataset(datasets)

    
    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader

def semisupervised_training(
    name: str,
    model,
    train_paths: Tuple[list[str], list[str]],
    val_paths: Tuple[list[str], list[str]],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: str,
    raw_key: str = "raw",
    load_path: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
) -> None:
    """Run semi-supervised segmentation training.

    Args:
        name: The name for the checkpoint to be trained.
        model: The model to be trained
        train_paths: tuple with labeled and unlabeled arrays of paths to HDF5 or zarr files for training 
        val_paths: tuple with labeled and unlabeled arrays of paths to HDF5 or zarr files for validation
        label_key: The key that holds the labels inside of the hdf5 or zarr.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5 or zarr.
        load_path: Filepath to the model to be trained. If None, will initialize from 0.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
    """
    # Loading of the previous model if load_path is not None
    if load_path is not None:
        if not os.path.exists(load_path):
            raise ValueError("Path to model is empty: " + load_path)
        model = load_model(load_path, model)

    # Keeping the separated paths for now; 
    
    train_loader = get_supervised_loader(train_paths[0], raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train)
    val_loader = get_supervised_loader(val_paths[0], raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val)

    unsupervised_train_loader = get_unsupervised_loader(train_paths[1], raw_key, patch_shape, batch_size,
                                                        n_samples_epoch=n_samples_train)
    unsupervised_val_loader = get_unsupervised_loader(val_paths[1], raw_key, patch_shape, batch_size,
                                                      n_samples_epoch=n_samples_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Self training functionality.
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=0.9)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
        reinit_teacher=load_path is None # Reinitialize only if we start from scratch
    )
    trainer.fit(n_iterations)
   
def weak_augmentations(p: float = 0.75) -> callable:
    """The weak augmentations used in the unsupervised data loader.

    Args:
        p: The probability for applying one of the augmentations.

    Returns:
        The transformation function applying the augmentation.
    """
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)

def get_supervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    add_boundary_transform: bool = True,
    label_dtype=torch.float32,
    rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[callable] = None,
    ignore_label: Optional[int] = None,
    label_transform: Optional[callable] = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for supervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 or zarr files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5 or zarr.
        label_key: The key that holds the labels inside of the hdf5 or zarr.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        add_boundary_transform: Whether to add a boundary channel to the training data.
        label_dtype: The datatype of the labels returned by the dataloader.
        rois: Optional region of interests for training.
        sampler: Optional sampler for selecting blocks for training.
            By default a minimum instance sampler will be used.
        ignore_label: Ignore label in the ground-truth. The areas marked by this label will be
            ignored in the loss computation. By default this option is not used.
        label_transform: Label transform that is applied to the segmentation to compute the targets.
            If no label transform is passed (the default) a boundary transform is used.
        loader_kwargs: Additional keyword arguments for the dataloader.

    Returns:
        The PyTorch dataloader.
    """

    if label_transform is not None:  # A specific label transform was passed, do nothing.
        pass
    elif add_boundary_transform:
        if ignore_label is None:
            label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
        else:
            label_transform = torch_em.transform.label.BoundaryTransformWithIgnoreLabel(
                add_binary_target=True, ignore_label=ignore_label
            )

    else:
        if ignore_label is not None:
            raise NotImplementedError
        label_transform = torch_em.transform.label.connected_components

    transform = torch_em.transform.Compose(
        torch_em.transform.PadIfNecessary(patch_shape), torch_em.transform.get_augmentations(3)
    )

    num_workers = loader_kwargs.pop("num_workers", 4 * batch_size)
    shuffle = loader_kwargs.pop("shuffle", True)

    if sampler is None:
        # sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)
        # This is the default, but will never work with this dataset; using the only value that works
        # (1) would make it useless.
        sampler = torch_em.data.sampler.MinForegroundSampler(0.01, 0.999)

    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=shuffle, n_samples=n_samples,
        label_dtype=label_dtype, rois=rois, **loader_kwargs,
    )
    return loader

if __name__ == "__main__":
    # Check label voxels - getting sampler timeout
    from model_utils import directory_to_path_list
    import h5py
    import numpy as np

    labeled_folder_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/target_labeled"
    source_labeled_folder_path = "/mnt/lustre-grete/usr/u15001/mitochondria/mitochondria/files/source_labeled"
    labeled_data_paths = directory_to_path_list(source_labeled_folder_path) + directory_to_path_list(labeled_folder_path) 
    for path in labeled_data_paths:
        with h5py.File(path) as f:
            data = f["label_crop/mito"]
            print(f"{path}: {np.count_nonzero(data[:])}")