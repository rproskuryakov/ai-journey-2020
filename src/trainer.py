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
            optimizer,
            device="cpu",
            n_epochs=100,
            metrics=None,
            scheduler=None,
            callbacks=None,
    ):
        self.net = network
        self.loss = loss
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.n_epochs = n_epochs
        self.net = self.net.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.metrics = metrics

    def fit(self):
        for n in range(self.n_epochs):
            logger.warning(f"Fitting {n} epoch...")
            self.training_step()
            self.validation_step()

    def training_step(self):
        train_loss = 0
        n_samples = 0

        logger.warning("Training")
        self.net.train()
        for idx, batch in enumerate(self.train_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            batch_size = len(batch_inputs)
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_input_lengths = batch_input_lengths.to(self.device)
            batch_label_lengths = batch_label_lengths.to(self.device)
            out = torch.reshape(self.net(batch_inputs), (512, 2, 77))
            loss = self.loss(
                out,
                batch_labels,
                batch_input_lengths,
                batch_label_lengths,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.detach() * batch_size
            n_samples += batch_size

        logger.warning("Train loss: %.2f" % (train_loss / n_samples))
        return train_loss / n_samples

    def validation_step(self):

        val_loss = 0
        n_samples = 0

        logger.warning("Evaluating")
        self.net.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
                batch_size = len(batch_inputs)
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_input_lengths = batch_input_lengths.to(self.device)
                batch_label_lengths = batch_label_lengths.to(self.device)
                out = torch.reshape(self.net(batch_inputs), (512, 2, 77))

                loss = self.loss(
                    out,
                    batch_labels,
                    batch_input_lengths,
                    batch_label_lengths,
                )
                val_loss += batch_size * loss.detach()
                n_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        logger.warning("Validation loss: %.2f" % (val_loss / n_samples))
        return val_loss / n_samples
