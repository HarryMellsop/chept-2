"""
Plot loss extracted from a ckpts directory.

Note: Run from project root. (torch.load requires model import structure
to be identical to when it was saved so this is a work around)

`python -m inference.plot_loss`

"""
import sys
sys.path.insert(1, '.')
import matplotlib.pyplot as plt
import numpy as np
import argparse
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

def get_just_epoch_num(ckpt_file):
    return int(ckpt_file.split("epoch_")[-1].split(".pt")[0])

def get_epoch_and_iter(ckpt_file):
    return get_epoch_num(ckpt_file), get_iteration_num(ckpt_file)

def get_sorted_checkpoints_by_epoch(ckpt_dir):
    
    files = glob.glob(ckpt_dir + "/epoch_[0-9].pt")
    epochs = [get_just_epoch_num(file) for file in files]

    sort_idx = np.argsort(epochs)
    files = np.array(files)[sort_idx].tolist()
    epochs = np.array(epochs)[sort_idx].tolist()

    return epochs, files

    

def get_sorted_checkpoints(ckpt_dir):
    """
    Returns checkpoints in increasing order of iteration.
    """

    files = glob.glob(ckpt_dir + "/epoch_*_iter_*.pt")
    iters = [get_epoch_and_iter(ckpt_file) for ckpt_file in files]

    sort_idx = sorted(range(len(iters)),key=iters.__getitem__)
    
    files = np.array(files)[np.array(sort_idx)].tolist()
    iters = np.array(iters)[np.array(sort_idx)].tolist()

    return iters, files

def get_losses(ckpt_dir, exclude_zeros=True, by_epoch=False):
    """"
    Returns loss values (along with iteration number)

    Kwargs are those for get_sorted_checkpoints.
    """

    if by_epoch:
        iters, files = get_sorted_checkpoints_by_epoch(ckpt_dir)
    else:
        iters, files = get_sorted_checkpoints(ckpt_dir)

    losses = []
    for file in tqdm.tqdm(files):

        # map to CPU cause we don't need CUDA here
        losses.append(torch.load(file, map_location="cpu")["loss"])

    losses = np.array(losses)
    if exclude_zeros:
        losses = losses[np.where(losses != 0)[0]]

    print(losses)

    return iters, losses

def plot_losses(iters, losses):
    """
    Plots losses vs. iteration
    """
    
    global_iters = [(epoch + 1)*iteration for (epoch, iteration) in iters]
    # global_iters = iters

    fig = plt.Figure()
    plt.plot(global_iters, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot loss from pt files.")
    parser.add_argument('--ckpt_dir', type=str, help='pt directory')
    args = parser.parse_args()

    iters, losses = get_losses(args.ckpt_dir, 
                               exclude_zeros=True,
                               by_epoch=False)
    plot_losses(iters, losses, by_epoch=False)
