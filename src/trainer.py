import logging
from typing import List
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.callbacks.base_callback import BaseCallback
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.save_checkpoints import SaveCheckpoints
from src.decoders.base_decoder import BaseDecoder
from src.metric_accamulator import MetricAccamulator
from src.metrics import CHARACTER_ERROR_RATE
from src.metrics import StringMetric

logger = logging.getLogger(__name__)


class TaskTrainer:
    def __init__(
            self,
            network: nn.Module,
            loss: nn.CTCLoss,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer,
            letters: List[str],
            device="cpu",
            n_epochs=100,
            metrics: Optional[List[StringMetric]] = None,
            scheduler: Optional = None,
            callbacks: Optional[List[BaseCallback]] = None,
            decoder: Optional[BaseDecoder] = None,
    ):
        self.letters = letters
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
        self.decoder = decoder
        self.metric_accamulator = MetricAccamulator()

    def fit(self):
        for n in range(self.n_epochs):
            logger.warning(f"Fitting {n} epoch...")
            self.training_step(n)
            self.validation_step(n)

    def training_step(self, epoch):
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

    def validation_step(self, epoch):

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
                out_val = None
                if self.decoder is not None:
                    out_val = self.decoder(out, self.letters)
                    batch_labels_decode = self.decoder.decode_labels(batch_labels, self.letters)
                    out_labels = self.decoder.decode_labels(out_val, self.letters)
                for metric in self.metrics:
                    if out_val is None:
                        raise ValueError('out_val must be init')
                    m = metric(pred_strings=out_labels, ground_strings=batch_labels_decode)
                    self.metric_accamulator.accumulate(m * batch_size, metric.name)
                loss = self.loss(
                    out,
                    batch_labels,
                    batch_input_lengths,
                    batch_label_lengths,
                )
                val_loss += batch_size * loss.detach()
                n_samples += batch_size

        cer = None
        for name, m in self.metric_accamulator.name_to_metric.items():
            logger.warning(f'{name}: ' + '%.2f' % (m / n_samples))
            if name == CHARACTER_ERROR_RATE:
                cer = m / n_samples

        for callback in self.callbacks:
            if isinstance(callback, SaveCheckpoints) and callback.only_best:
                if cer is None:
                    raise ValueError('cer must be init')
                callback(n_epoch=epoch, metric=cer)
            if isinstance(callback, EarlyStopping):
                callback(val_loss / n_samples)

        if self.scheduler is not None:
            self.scheduler.step()

        logger.warning("Validation loss: %.2f" % (val_loss / n_samples))
        return val_loss / n_samples
