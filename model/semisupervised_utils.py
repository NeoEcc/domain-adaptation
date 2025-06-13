# First functions adapted from synapse-net to work with zarrs
# https://github.com/computational-cell-analytics/synapse-net/blob/main/synapse_net/training/semisupervised_training.py

from typing import Optional, Tuple

import numpy as np
import torch
import torch_em
import torch_em.self_training as self_training
from torchvision import transforms
import random


def get_unsupervised_loader(
    data_path: str,
    raw_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples_epoch: Optional[int],
    roi = None,
    blacklist_roi = None,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_path: The filepath to the hdf5 or zarr file from which to sample the training data.
        raw_key: The key that holds the raw data inside of the hdf5 or zarr.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        n_samples_epoch: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        roi: specify a region of interest, can be None.
        blacklist_roi: list of regions to be avoided, as array of tuples of slices, or None

    Returns:
        The PyTorch dataloader.
    """

    ndim = 3

    # Transforms and augmentations definition
    # TODO: implement strong augmentations
    raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=ndim)
    augmentations = (weak_augmentations(), weak_augmentations())
    
    if n_samples_epoch is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples_epoch / len(datasets))
    
    ### MODIFIED HERE TO ADAPT TO ZARR  
    datasets = [
        torch_em.data.RawDataset(data_path, raw_key, patch_shape, raw_transform, transform,
                                 augmentations=augmentations, roi=get_random_roi(roi, patch_shape, blacklist_roi),
                                 ndim=ndim, n_samples=n_samples_per_ds)
        for _ in range(n_samples_epoch)
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    
    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader

def semisupervised_training(
    name: str,
    model,
    unlabeled_train_path: str,
    labeled_train_path: str,
    val_paths: Tuple[str],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: str,
    raw_key: str = "raw",
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
        train_paths: Filepath to the hdf5 or zarr file for the training data.
        val_paths: Filepaths to the hdf5 or zarr files for the validation data.
        label_key: The key that holds the labels inside of the hdf5 or zarr.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5 or zarr.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
    """
    # Keeping the separated paths for now; 
    
    train_loader = get_supervised_loader(labeled_train_path, raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val)

    unsupervised_train_loader = get_unsupervised_loader(unlabeled_train_path, raw_key, patch_shape, batch_size,
                                                        n_samples_epoch=n_samples_train)
    unsupervised_val_loader = get_unsupervised_loader(val_paths, raw_key, patch_shape, batch_size,
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
        sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)

    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=shuffle, n_samples=n_samples,
        label_dtype=label_dtype, rois=rois, **loader_kwargs,
    )
    return loader

def slices_overlap(a, b):
    # Returns True if slices a and b overlap in any dimensions
    x = []
    for d in range(len(a)):
        x.append(a[d].stop < b[d].start or a[d].start >= b[d].stop) 
    return not all(x) # Can be probably made easier with a not and a direct return true

def get_random_roi(original_roi, size, blacklist = None):
    """
    Returns a random region of interest within the original roi coordinate space
    that does not include any voxel in the blacklist, of the specified size. 
    Works in any dimension. Boundaries of the blacklist are also excluded.

    Algorithm, to be used dimension-wide:
    - Get a starting point between beginning and end - size
    - Create final point adding size to starting point
    - For all items in blacklist:
        - If the initial point of the roi is larger than the final point, all good
        - If the final point is smaller than the initial point, all good.
    - If these conditions are met, return a valid roi.
    - Otherwise, add 1 to the counter and restart. 
    - If the counter reaches a threshold, stop. Could be extremely unlucky or wrong initialization.
        
    Args:
        original_roi: The original region of interest within which to sample.
        size: The shape (tuple of ints) of the desired random ROI.
        blacklist: A list of regions to avoid when sampling the ROI. None to accept all samples. 
        Must be an array of tuples of slices.
    """
    max_attempts = 250

    # Input checks
    if original_roi is None:
        raise ValueError("original_roi must be provided")
    if isinstance(original_roi, tuple):
        roi_slices = original_roi
    else:
        roi_slices = tuple(original_roi)

    # If there is some none, set to 0 - makes sense only in the start
    starts = [slc.start if slc.start is not None else 0 for slc in roi_slices]
    stops = [slc.stop for slc in roi_slices]
    dims = len(starts)

    # Get the maximum starting point between start and end - size
    valid_starts = []
    for d in range(dims):
        max_start = stops[d] - size[d]
        if max_start < starts[d]:
            raise ValueError(f"ROI size {size} is too large for dimension {d} in original_roi {original_roi}")
        valid_starts.append((starts[d], max_start))

    # Try up to max_attempts times to find a non-blacklisted ROI
    for _ in range(max_attempts):
        rand_start = [random.randint(valid_starts[d][0], valid_starts[d][1]) for d in range(dims)]
        rand_stop = [rand_start[d] + size[d] for d in range(dims)]
        candidate = tuple(slice(rand_start[d], rand_stop[d]) for d in range(dims))

        # Check overlap
        overlap = False
        if blacklist is not None:
            for bl in blacklist:
                if slices_overlap(candidate, bl):
                    overlap = True
                    break
                
        if not overlap:
            # assert not any(slices_overlap(candidate, bl) for bl in blacklist), f"Overlap detected: {candidate} vs {blacklist}"
            return candidate

    raise RuntimeError(f"Could not find a valid ROI after {max_attempts} attempts")


if __name__ == "__main__":
    # Test for ROI
    original_roi = slice(0, 100, 1), slice(0, 100, 1), slice(0, 100, 1)
    size = (20, 20, 20)
    blacklist = [(slice(10, 30, 1), slice(40, 60, 1), slice(70, 90, 1))]

    # for x in range(5):
    #     res = get_random_roi(original_roi, size, blacklist)
    #     print(f"Random ROI in {original_roi}: {res}")

    # Test dataset
    n_crops = 2
    data_path = "/user/niccolo.eccel/u15001/example_dataset/jrc_ctl-id8-2.zarr"
    raw_key = "/recon-1/em/fibsem-uint8/s0"
    patch_shape = (5,)*3
    raw_transform = None
    transform = None
    augmentations = None
    raw_roi = (
        slice(10000, 30000, 1),
        slice(1000, 9000, 1),
        slice(5000, 60000, 1)
    )
    blacklist_roi = [(slice(10949, 11349), slice(6089, 9000), slice(19209, 19609))]
    small_roi=get_random_roi(raw_roi, patch_shape, blacklist_roi)
    print(small_roi)

    datasets = [
        torch_em.data.RawDataset(data_path, raw_key, patch_shape, raw_transform, transform,
                                    augmentations=augmentations, roi=small_roi,
                                    ndim=3, n_samples=30)
        for _ in range(n_crops)
    ]
    print(datasets[0])
    print(type(datasets[0]))
    print(datasets[0].__getitem__(1))
    print(len(datasets))
    print(type(datasets[0].__getitem__(0)[0][0][0][0]))