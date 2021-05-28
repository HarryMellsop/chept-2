import random
import numpy as np
import torch
from torch.nn import functional as F
import json

default_config_args = {
    "block_size": 256,
    "n_layer": 12,
    "n_head": 32,
    "n_embed": 256
}

default_train_args = {
    "max_epochs": 1,
    "batch_size": 10,
    "grad_norm_clip": 1.0,
    "learning_rate": 1e-3,
    "num_workers": 4,
    "save_interval": 1000,
}


class TrainArgs:
    """
    Class to handle creation of arguments for training. Handles pretraining and finetunring scenarios.
    """

    def __init__(self, args_path, provided_args, pretrain_args=None, default_config=default_config_args, default_train=default_train_args):

        self.update_defaults(default_config, default_train)
        if pretrain_args:
            self.__dict__.update(pretrain_args)
        self.file_update(args_path)
        self.provided_update(provided_args)

    def update_defaults(self, default_config, default_train):

        self.__dict__.update(default_config)
        self.__dict__.update(default_train)

    def file_update(self, args_path):

        if args_path:
            with open(args_path) as f:
                data = json.load(f)
                self.__dict__.update(data)

    def provided_update(self, provided_args):

        # ensure we aren't updating null vals
        for key, val in provided_args.items():
            if val:
                self.__dict__[key] = val

    def __call__(self):

        config_args = {key: val for key, val in self.__dict__.items() if key in default_config_args}
        train_args = {key: val for key, val in self.__dict__.items() if key in default_train_args}

        return config_args, train_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

# Get the most recent parameters file from the ckpts
def get_recent_ckpt(ckpt_dir):

    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"Default checkpoint dir at {ckpt_dir} missing!")

    files = os.listdir(ckpt_dir)
    if 'best_loss.pt' in files:
        answer = questionary.confirm("File best_loss.pt found. Use this file?").ask()
        if answer:
            return os.path.join(ckpt_dir, 'best_loss.pt')
    epoch_list = [x for x in files if 'epoch' in x]
    if len(epoch_list) > 0:
        answer = questionary.confirm("Epoch files found. Use best epoch file?").ask()
        if answer:
            epoch_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            return os.path.join(ckpt_dir, epoch_list[0])

    iter_list = [x for x in files if 'iter' in x]
    iter_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)

    return os.path.join(ckpt_dir, iter_list[0])