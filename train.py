from functools import reduce
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as tf

from src.models import ResNet18AttentionNetwork
from src.models.resnet_model import ResNet18Network
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
    LEARNING_RATE = 0.001
    N_EPOCHS = 200
    BATCH_SIZE = 10

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    logger.info(f"Device: {device}")
    dataframe = pd.read_csv("data/interim/texts.csv")
    filenames = dataframe["filename"].to_list()
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2, shuffle=True, random_state=2020)
    MAX_LEN = dataframe["text"].str.len().max()
    letters = ["_"] + list(reduce(lambda x, y: set(x) | set(y), dataframe["text"].to_list(), set()))

    transformer = tf.Compose([
        tf.Resize((128, 1024)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = PetrDataset(
        filenames=train_filenames,
        image_folder=INPUT_DIR / "train/images",
        text_folder=INPUT_DIR / "train/words",
        letters=letters,
        max_len=MAX_LEN,
        transformer=transformer
    )

    val_dataset = PetrDataset(
        filenames=test_filenames,
        image_folder=INPUT_DIR / "train/images",
        text_folder=INPUT_DIR / "train/words",
        letters=letters,
        max_len=MAX_LEN,
        transformer=transformer
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    network = ResNet18AttentionNetwork(n_letters=len(letters))

    lm_optimizer = Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        lm_optimizer,
        mode="min",
        patience=5,
        factor=0.7
    )

    trainer = TaskTrainer(
        network=network,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss=nn.CTCLoss(blank=0, reduction="mean"),
        optimizer=lm_optimizer,
        device=device,
        n_epochs=N_EPOCHS,
        scheduler=scheduler,
        callbacks=[
            EarlyStopping(patience=20, min_delta=1e-5),
            SaveCheckpoints(network, only_best=True, folder=OUTPUT_DIR / 'checkpoint_torch/v3_resnet_attention/'),
        ],
        metrics=[
            WordErrorRate(),
            CharacterErrorRate(),
            StringAccuracy(),
        ],
        letters=letters,
        decoder=GreedyDecoder(letters, blank_id=0),
        clipping_value=10
    )
    trainer.fit()
