from functools import reduce
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data

from src.models.baseline_model import BaselineNetwork
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.save_checkpoints import SaveCheckpoints
from src.dataset import PetrDataset
from src.dataset import BaselineTransformer
from src.decoders.greedy_decoder import GreedyDecoder
from src.metrics import CharacterErrorRate
from src.metrics import StringAccuracy
from src.metrics import WordErrorRate
from src.trainer import TaskTrainer

logging.basicConfig(
    # filename="log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Launch parameters
    INPUT_DIR = Path("data/")
    OUTPUT_DIR = Path("models/")
    LEARNING_RATE = 0.01
    N_EPOCHS = 1
    BATCH_SIZE = 10

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    logger.info(f"Device: {device}")
    dataframe = pd.read_csv("data/interim/texts.csv")[:50]
    filenames = dataframe["filename"].to_list()
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, shuffle=True, random_state=2020)
    MAX_LEN = dataframe["text"].str.len().max()
    letters = ["_"] + list(reduce(lambda x, y: set(x) | set(y), dataframe["text"].to_list(), set()))

    transformer = BaselineTransformer()
    train_dataset = PetrDataset(
        filenames=train_filenames,
        image_folder=INPUT_DIR / "train/images",
        text_folder=INPUT_DIR / "train/words",
        letters=letters,
        max_len=MAX_LEN,
        transformer=transformer,
    )

    val_dataset = PetrDataset(
        filenames=test_filenames,
        image_folder=INPUT_DIR / "train/images",
        text_folder=INPUT_DIR / "train/words",
        letters=letters,
        max_len=MAX_LEN,
        transformer=transformer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    network = BaselineNetwork(n_letters=len(letters))

    optimizer = AdamW(network.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_dataloader),
        epochs=N_EPOCHS,
    )

    trainer = TaskTrainer(
        network=network,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss=nn.CTCLoss(blank=0),
        optimizer=optimizer,
        device=device,
        n_epochs=N_EPOCHS,
        scheduler=scheduler,
        callbacks=[
            EarlyStopping(patience=20, min_delta=1.e-5),
            SaveCheckpoints(network, only_best=True, folder=OUTPUT_DIR / 'checkpoint_torch/v1/'),
        ],
        metrics=[
            WordErrorRate(),
            CharacterErrorRate(),
            StringAccuracy(),
        ],
        letters=letters,
        decoder=GreedyDecoder(letters=letters, blank_id=0, log_probs_input=True)
    )
    trainer.fit()
