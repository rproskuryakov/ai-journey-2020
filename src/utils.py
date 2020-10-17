import pathlib

import torch


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.val_losses = []
        self.n_epochs_without_improving = 0

    def __call__(self, val_loss):
        if (self.val_losses[-1] - val_loss) <= self.min_delta:
            self.n_epochs_without_improving += 1
        else:
            self.n_epochs_without_improving = 0

        return self.n_epochs_without_improving != self.patience


class SaveCheckpoints:
    def __init__(self, model, folder):
        self.folder = folder
        self.model = model

    def save_checkpoint(self, n_epoch):
        path = pathlib.Path(self.folder) / pathlib.Path(f'model_epoch_{n_epoch}.pth')
        torch.save(self.model.cpu().state_dict(), path)
        self.model.cuda()
