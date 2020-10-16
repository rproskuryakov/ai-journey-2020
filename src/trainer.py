import logging

import torch.nn as nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class TaskTrainer:
    def __init__(self, network: nn.Module, loss: nn.CTCLoss, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 device="cpu", n_epochs=100, metrics=None):
        self.net = network
        self.loss = loss
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.n_epochs = n_epochs

    def fit(self):
        for n in range(self.n_epochs):
            logger.info(f"Fitting {n} epoch...")
            self.fit_epoch()

    def fit_epoch(self):
        train_loss = 0
        val_loss = 0

        self.net.train()
        for idx, batch in enumerate(self.train_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            loss = self.loss(self.net.forward(batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths))
            loss.backward()
            train_loss += loss.detach()

        logger.info("Train loss: %.2f" % train_loss)

        self.net.eval()
        for idx, batch in enumerate(self.val_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            loss = self.loss(self.net.forward(batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths))
            val_loss += loss.detach()

        logger.info("Validation loss: %.2f" % val_loss)
