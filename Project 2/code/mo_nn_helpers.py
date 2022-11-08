"""
Maximilian Otto, 2022, maxotto45@gmail.com
Utility functions for training and evaluating models.
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from tqdm import tqdm


def try_make_dir(d):
    '''
    Create Directory-path, if it doesn't exist yet.
    '''
    import os
    if not os.path.isdir(d):
        os.makedirs(d)
    return

def get_mean_and_std(data_dir):
    '''
    Acquire the mean and std color values of all images (RGB-values) in the training set.
    inpupt: "data_dir" string
    output: mean and std Tensors of size 3 (RGB)
    '''
    # Load the training set
    train_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transforms.ToTensor()) for x in ["train"]}
    train_loader = {x: torch.utils.data.DataLoader(dataset=train_dataset[x], batch_size=1, num_workers=0) for x in ["train"]}
    # Calculate the mean and std of the training set
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(train_loader["train"], desc="Calculating mean and std of all RGB-values"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    #var[x] = E[x**2] - E[X]**2
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("Mean: ", mean, ", Std: ", std)
    return mean, std