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
    return int(ckpt_file.split("epoch_")[-1].split("_iter_")[-1].split(".pt")[0])

def get_epoch_num(ckpt_file):
    """
    Returns the epoch number from a ckpt_file or path.
    """
    return int(ckpt_file.split("epoch_")[-1].split("_iter")[0])

def get_epoch_and_iter(ckpt_file):
    return get_epoch_num(ckpt_file), get_iteration_num(ckpt_file)
    

def get_sorted_checkpoints(ckpt_dir, exclude_first=True):
    """
    Returns checkpoints in increasing order of iteration.
    """

    files = glob.glob(ckpt_dir + "/epoch_*_iter_*.pt")
    iters = [get_epoch_and_iter(ckpt_file) for ckpt_file in files]

    sort_idx = sorted(range(len(iters)),key=iters.__getitem__)
    
    files = np.array(files)[np.array(sort_idx)].tolist()

    if exclude_first:
        files = files[1:]

    return iters, files

def get_losses(ckpt_dir, exclude_first=True):
    """"
    Returns loss values (along with iteration number)

    Kwargs are those for get_sorted_checkpoints.
    """

    iters, files = get_sorted_checkpoints(ckpt_dir, 
                                          exclude_first=exclude_first)

    losses = []
    for file in tqdm.tqdm(files):

        # map to CPU cause we don't need CUDA here
        losses.append(torch.load(file, map_location="cpu")["loss"])

    return iters.tolist(), losses

def plot_losses(iters, losses):
    """
    Plots losses vs. iteration
    """

    global_iters = [(epoch + 1)*iteration for (epoch, iteration) in iters]

    fig = plt.Figure()
    plt.plot(global_iters, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.show()


if __name__ == "__main__":
    ckpt_dir = "/media/schlager/COLLIN/finetune_default_no_mask"
    iters, losses = get_losses(ckpt_dir, exclude_first=False)
    plot_losses(iters, losses)