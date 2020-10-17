from functools import reduce

import pandas as pd
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

from src.baseline_model import BaselineNetwork
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.save_checkpoints import SaveCheckpoints
from src.dataset import PetrDataset
from src.decoders.ctc_decoder import CTCDecoder
from src.metrics import CharacterErrorRate
from src.metrics import StringAccuracy
from src.metrics import WordErrorRate
from src.trainer import TaskTrainer

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    dataframe = pd.read_csv("data/interim/texts.csv")
    filenames = dataframe["filename"].to_list()
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.15, shuffle=True, random_state=2020)
    MAX_LEN = dataframe["text"].str.len().max()
    letters = list(reduce(lambda x, y: set(x) | set(y), dataframe["text"].to_list(), set()))

    train_dataset = PetrDataset(filenames=train_filenames,
                                image_folder="data/train/images",
                                text_folder="data/train/words",
                                letters=letters,
                                max_len=MAX_LEN)

    val_dataset = PetrDataset(filenames=test_filenames,
                              image_folder="data/train/images",
                              text_folder="data/train/words",
                              letters=letters,
                              max_len=MAX_LEN)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    network = BaselineNetwork(n_letters=len(letters))

    optimizer = AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    trainer = TaskTrainer(
        network=network,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss=nn.CTCLoss(),
        optimizer=optimizer,
        device=device,
        n_epochs=20,
        scheduler=scheduler,
        callbacks=[
            EarlyStopping(patience=20, min_delta=1.e-5),
            SaveCheckpoints(network, only_best=True, folder='checkpoint_torch/v1/'),
        ],
        metrics=[
            WordErrorRate(),
            CharacterErrorRate(),
            StringAccuracy(),
        ],
        letters=letters,
        decoder=CTCDecoder()
    )
    trainer.fit()
