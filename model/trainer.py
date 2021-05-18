from tqdm import tqdm
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


class TrainerConfig:

    def __init__(self, func, state_dict, args_dict):

        # update new params
        self.func = func
        self.state_dict = state_dict
        self.__dict__.update(args_dict)


class Trainer:

    def __init__(self, model, train_dataset, save_dir, config):
        self.model = model
        self.model_config = model.model_config
        self.train_dataset = train_dataset
        self.config = config
        self.func = self.config.func
        self.load_params = self.config.state_dict

        self.save_dir = save_dir

        if self.load_params:
        # ckpt loading
            print('\nLoading params...\n')
            self.model.load_state_dict(self.load_params)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, path, loss=None):
        save_path = os.path.join(self.save_dir, path)
        ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
        save_dict = {'state_dict': ckpt_model.state_dict(),
                     'itos': self.train_dataset.vocab.itos,
                     'stoi': self.train_dataset.vocab.stoi,
                     'model_config': self.model_config,
                     'train_config': self.config,
                     'loss': loss}

        torch.save(save_dict, save_path)

    def train(self):
        model, config = self.model, self.config

        print(config.batch_size)

        # create the optimizer
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate,
        )

        def run_epoch(split):
            model.train(True)

            data = self.train_dataset
            loader = DataLoader(data, 
                batch_size=config.batch_size, 
                num_workers=config.num_workers
            )

            min_loss = float('inf')
            pbar = tqdm(enumerate(loader), total=len(loader))
            cumulative_loss = 0
            for it, (x, y) in pbar:
                if it % config.save_interval == 0:
                    avg_loss = cumulative_loss / config.save_interval
                    self.save_checkpoint(f'iter_{it}.pt', loss=avg_loss)
                    cumulative_loss = 0

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(x, y)
                    loss = loss.mean()
                if loss <= (0.8 * min_loss):
                    min_loss = loss
                    self.save_checkpoint('best_loss.pt', loss=loss.item())

                # accumulate avg loss for readout
                cumulative_loss += loss.item()

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch('train')
            self.save_checkpoint(f'epoch_{epoch}.pt')