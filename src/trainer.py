import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TaskTrainer:
    def __init__(
            self,
            network: nn.Module,
            loss: nn.CTCLoss,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            device="cpu",
            n_epochs=100,
            metrics=None,
    ):
        self.net = network
        self.loss = loss
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.n_epochs = n_epochs
        self.net = self.net.to(self.device)

    def fit(self):
        for n in range(self.n_epochs):
            logger.warning(f"Fitting {n} epoch...")
            self.fit_epoch()

    def fit_epoch(self):
        train_loss = 0
        val_loss = 0

        logger.warning("Training")
        self.net.train()
        for idx, batch in enumerate(self.train_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            # batch_input_lengths = batch_input_lengths.squeeze()
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_input_lengths = batch_input_lengths.to(self.device)
            batch_label_lengths = batch_label_lengths.to(self.device)
            out = torch.reshape(self.net.forward(batch_inputs), (512, 2, 77))
            # out = out.resize(255, 2, 77)
            loss = self.loss(
                out,
                batch_labels,
                batch_input_lengths,
                batch_label_lengths,
            )
            loss.backward()
            train_loss += loss.detach()

        logger.warning("Train loss: %.2f" % train_loss)

        logger.warning("Evaluating")
        self.net.eval()
        for idx, batch in enumerate(self.val_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_input_lengths = batch_input_lengths.to(self.device)
            batch_label_lengths = batch_label_lengths.to(self.device)
            out = torch.reshape(self.net.forward(batch_inputs), (512, 2, 77))

            loss = self.loss(
                out,
                batch_labels,
                batch_input_lengths,
                batch_label_lengths,
            )
            val_loss += loss.detach()

        logger.warning("Validation loss: %.2f" % val_loss)
