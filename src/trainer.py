import logging
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.callbacks.base_callback import BaseCallback
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.save_checkpoints import SaveCheckpoints
from src.decoders.base_decoder import BaseDecoder
from src.metric_accumulator import MetricAccumulator
from src.metrics import StringMetric

logger = logging.getLogger(__name__)


class TaskTrainer:
    def __init__(
            self,
            network: nn.Module,
            loss,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            optimizer,
            letters: List[str],
            clipping_value: float,
            device="cpu",
            n_epochs=100,
            metrics: Optional[List[StringMetric]] = None,
            scheduler: Optional = None,
            callbacks: Optional[List[BaseCallback]] = None,
            decoder: Optional[BaseDecoder] = None,
    ):
        self.letters = letters
        self.net = network
        self.loss = loss.to(device)
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
        self.clipping_value = clipping_value
        self.metric_accumulator = MetricAccumulator()

    def fit(self):
        for n in range(1, self.n_epochs + 1):
            logger.info(f"Fitting {n} epoch...")
            print(f"Fitting {n} epoch...")
            self.training_step(n)
            val_loss = self.validation_step(n)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

    def training_step(self, epoch):
        train_loss = 0
        n_samples = 0

        logger.info("Training...")
        print("Training...")
        self.net.train()
        for idx, batch in enumerate(self.train_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            batch_size = len(batch_inputs)
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_input_lengths = batch_input_lengths.to(self.device)
            batch_label_lengths = batch_label_lengths.to(self.device)
            out = self.net(batch_inputs)
            loss = self.loss(
                out,
                batch_labels,
                batch_input_lengths,
                batch_label_lengths,
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipping_value)
            self.optimizer.step()
            train_loss += loss.detach() * batch_size
            n_samples += batch_size

        logger.info("Train loss: %.2f" % (train_loss / n_samples))
        print("Train loss: %.2f" % (train_loss / n_samples))
        return train_loss / n_samples

    def validation_step(self, epoch):

        val_loss = 0
        n_samples = 0

        logger.info("Evaluating")
        print("Evaluating")
        self.net.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
                batch_size = len(batch_inputs)
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_input_lengths = batch_input_lengths.to(self.device)
                batch_label_lengths = batch_label_lengths.to(self.device)
                out = self.net(batch_inputs)
                # OUT_SHAPE: (T, N, C)
                out_val = None
                if self.decoder is not None:
                    out_val = self.decoder(out)
                    pred_texts, true_texts = self.decoder.decode(out_val, batch_labels, batch_label_lengths)
                for metric in self.metrics:
                    if out_val is None:
                        raise ValueError('out_val must be init')
                    metric(pred_strings=pred_texts, ground_strings=true_texts)

                loss = self.loss(
                    out,
                    batch_labels,
                    batch_input_lengths,
                    batch_label_lengths,
                )
                val_loss += batch_size * loss.detach()
                n_samples += batch_size

        cer = None
        for m in self.metrics:
            value = m.calculate()
            logger.info(f'{m.name()}: %.2f' % value)
            print(f'{m.name()}: ' + '%.2f' % value)
            if m.name() == "CharacterErrorRate":
                cer = value

        for callback in self.callbacks:
            if isinstance(callback, SaveCheckpoints) and callback.only_best:
                if cer is None:
                    raise ValueError('cer must be init')
                callback(n_epoch=epoch, metric=cer)
            if isinstance(callback, EarlyStopping):
                callback(val_loss / n_samples)

        logger.info("Validation loss: %.2f" % (val_loss / n_samples))
        print("Validation loss: %.2f" % (val_loss / n_samples))
        return val_loss / n_samples


class TransformerTrainer:
    def __init__(
            self,
            network: nn.Module,
            loss,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            optimizer,
            letters: List[str],
            clipping_value: float,
            device="cpu",
            n_epochs=100,
            metrics: Optional[List[StringMetric]] = None,
            scheduler: Optional = None,
            callbacks: Optional[List[BaseCallback]] = None,
            decoder: Optional[BaseDecoder] = None,
    ):
        self.letters = letters
        self.net = network
        self.loss = loss.to(device)
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
        self.clipping_value = clipping_value
        self.metric_accumulator = MetricAccumulator()

    def fit(self):
        for n in range(1, self.n_epochs + 1):
            logger.info(f"Fitting {n} epoch...")
            print(f"Fitting {n} epoch...")
            self.training_step(n)
            val_loss = self.validation_step(n)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

    def training_step(self, epoch):
        train_loss = 0
        n_samples = 0

        logger.info("Training...")
        print("Training...")
        self.net.train()
        for idx, batch in enumerate(self.train_loader):
            batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
            batch_size = len(batch_inputs)
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_input_lengths = batch_input_lengths.to(self.device)
            batch_label_lengths = batch_label_lengths.to(self.device)
            out = self.net(batch_inputs)
            loss = self.loss(
                out,
                batch_labels,
                batch_input_lengths,
                batch_label_lengths,
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipping_value)
            self.optimizer.step()
            train_loss += loss.detach() * batch_size
            n_samples += batch_size

        logger.info("Train loss: %.2f" % (train_loss / n_samples))
        print("Train loss: %.2f" % (train_loss / n_samples))
        return train_loss / n_samples

    def validation_step(self, epoch):

        val_loss = 0
        n_samples = 0

        logger.info("Evaluating")
        print("Evaluating")
        self.net.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                batch_inputs, batch_labels, batch_input_lengths, batch_label_lengths = batch
                batch_size = len(batch_inputs)
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_input_lengths = batch_input_lengths.to(self.device)
                batch_label_lengths = batch_label_lengths.to(self.device)
                out = self.net(batch_inputs)
                # OUT_SHAPE: (T, N, C)
                out_val = None
                if self.decoder is not None:
                    out_val = self.decoder(out)
                    pred_texts, true_texts = self.decoder.decode(out_val, batch_labels, batch_label_lengths)
                for metric in self.metrics:
                    if out_val is None:
                        raise ValueError('out_val must be init')
                    metric(pred_strings=pred_texts, ground_strings=true_texts)

                loss = self.loss(
                    out,
                    batch_labels,
                    batch_input_lengths,
                    batch_label_lengths,
                )
                val_loss += batch_size * loss.detach()
                n_samples += batch_size

        cer = None
        for m in self.metrics:
            value = m.calculate()
            logger.info(f'{m.name()}: %.2f' % value)
            print(f'{m.name()}: ' + '%.2f' % value)
            if m.name() == "CharacterErrorRate":
                cer = value

        for callback in self.callbacks:
            if isinstance(callback, SaveCheckpoints) and callback.only_best:
                if cer is None:
                    raise ValueError('cer must be init')
                callback(n_epoch=epoch, metric=cer)
            if isinstance(callback, EarlyStopping):
                callback(val_loss / n_samples)

        logger.info("Validation loss: %.2f" % (val_loss / n_samples))
        print("Validation loss: %.2f" % (val_loss / n_samples))
        return val_loss / n_samples