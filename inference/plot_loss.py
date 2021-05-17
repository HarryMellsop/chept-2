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

# import pdb; pdb.set_trace()
# sys.path.append(os.getcwd().split("/")[:-1])

def get_iteration_num(ckpt_file):
    """
    Returns the iteration number from a ckpt_file or path.
    """
    return int(ckpt_file.split("iter_")[-1].split(".pt")[0])

def get_sorted_checkpoints(ckpt_dir, exclude_first=True):
    """
    Returns checkpoints in increasing order of iteration.
    """

    files = np.array(glob.glob(ckpt_dir + "/iter_*.pt"))
    iters = np.array([get_iteration_num(file) for file in files])
    
    sorted_idx = np.argsort(iters)
    files = files[sorted_idx]

    if exclude_first:
        files = files[1:]

    return files

def get_losses(ckpt_dir):

    files = get_sorted_checkpoints(ckpt_dir)

    iters = []
    losses = []
    for file in tqdm.tqdm(files):
        iters.append(get_iteration_num(file))
        losses.append(torch.load(file)["loss"])

    return iters, losses

def plot_losses(iters, losses):

    fig = plt.Figure()
    plt.plot(iters, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.show()


if __name__ == "__main__":
    ckpt_dir = "ckpts/pretrain_default"
    iters, losses = get_losses(ckpt_dir)
    plot_losses(iters, losses)