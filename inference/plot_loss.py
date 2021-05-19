"""
Plot loss extracted from a ckpts directory.

Note: Run from project root. (torch.load requires model import structure
to be identical to when it was saved so this is a work around)

`python -m inference.plot_loss`

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import tqdm
import sys
import os


def get_iteration_num(ckpt_file):
    """
    Returns the iteration number from a ckpt_file or path.
    """
    return int(ckpt_file.split("iter_")[-1].split(".pt")[0])

def get_epoch_num(ckpt_file):
    """
    Returns the epoch number from a ckpt_file or path.
    """
    return int(ckpt_file.split("epoch_")[-1].split(".pt")[0])

def get_sorted_checkpoints(ckpt_dir, exclude_first=True, by_epoch=False):
    """
    Returns checkpoints in increasing order of iteration.

    if by_epoch: only returns epoch checkpoints
    """

    if by_epoch:
        match_str = "/epoch_*.pt"
        extract_fn = get_epoch_num
    else:
        match_str = "/iter_*.pt"
        extract_fn = get_iteration_num

    files = np.array(glob.glob(ckpt_dir + match_str))
    iters = np.array([extract_fn(file) for file in files])
    
    sorted_idx = np.argsort(iters)
    files = files[sorted_idx]

    if exclude_first:
        files = files[1:]

    return iters, files

def get_losses(ckpt_dir, by_epoch=False, exclude_first=True):
    """"
    Returns loss values (along with iteration number)

    Kwargs are those for get_sorted_checkpoints.
    """

    iters, files = get_sorted_checkpoints(ckpt_dir, 
                                          exclude_first=exclude_first, 
                                          by_epoch=by_epoch)

    losses = []
    for file in tqdm.tqdm(files):

        # map to CPU cause we don't need CUDA here
        losses.append(torch.load(file, map_location="cpu")["loss"])

    return iters.tolist(), losses

def plot_losses(iters, losses):
    """
    Plots losses vs. iteration
    """

    fig = plt.Figure()
    plt.plot(iters, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.show()


if __name__ == "__main__":
    ckpt_dir = "ckpts/pretrain_default_2"
    iters, losses = get_losses(ckpt_dir, exclude_first=False, by_epoch=True)
    plot_losses(iters, losses)