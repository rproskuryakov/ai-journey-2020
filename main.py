from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.data

from src.dataset import PetrDataset
from src.baseline_model import BaselineNetwork
from src.trainer import TaskTrainer

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    dataframe = pd.read_csv("data/interim/texts.csv")
    filenames = dataframe["filename"]
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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    network = BaselineNetwork(n_letters=len(letters))

    trainer = TaskTrainer(network=network,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          loss=nn.CTCLoss(),
                          device=device,
                          n_epochs=20)
    trainer.fit()
