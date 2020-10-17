from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.data
from torch.optim import AdamW

from src.dataset import PetrDataset
from src.baseline_model import BaselineNetwork
from src.trainer import TaskTrainer
from src.utils import EarlyStopping, SaveCheckpoints


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # torch.cuda.set_device('0')
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
        callbacks=[EarlyStopping(patience=20, min_delta=0.5), SaveCheckpoints()]
    )
    trainer.fit()
