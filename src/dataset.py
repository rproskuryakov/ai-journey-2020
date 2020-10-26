from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaselineTransformer:
    def __init__(self):
        pass

    def __call__(self, img):
        w, h = img.size
        new_h = 128
        new_w = int(w * (new_h / h))

        img = img.resize((new_w, new_h))
        w, h = img.size

        img_array = np.array(img)

        if h < 128:
            add_zeros = np.full((128 - h, w, 3), 255)
            img_array = np.concatenate((img_array, add_zeros))
            h, w, _ = img_array.shape

        if w < 1024:
            add_zeros = np.full((h, 1024 - w, 3), 255)
            img_array = np.concatenate((img_array, add_zeros), axis=1)
            h, w, _ = img_array.shape

        if w > 1024 or h > 128:
            dim = (128, 1024)
            img_array = np.array(Image.fromarray(img_array).resize(dim)).reshape((128, 1024, 3))

        img_array = 255 - img_array

        img_array = img_array / 255

        img_array = img_array.reshape(3, 128, 1024)

        return torch.from_numpy(np.array(img_array).astype(dtype="float32"))


class PetrDataset(Dataset):

    def __init__(self, filenames, image_folder, text_folder, letters, max_len=0, transformer=None):
        self.filenames = filenames
        self.image_folder = Path(image_folder)
        self.text_folder = Path(text_folder)
        self.max_len = max_len
        self.letters = letters
        self.transformer = transformer

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = Path(self.filenames[idx])

        # text processing
        with open(self.text_folder / filename, "r") as file:
            text = file.read().strip()
            encoded_text = self.text_to_label(text)
            padded_text = self.pad_sequence(encoded_text, value=self.letters.index(" "))

        # image processing
        image = Image.open(self.image_folder / filename.with_suffix(".jpg"))
        image = self.transformer(image)

        return (
            image,
            torch.Tensor(padded_text),
            255,
            len(encoded_text),
        )

    def text_to_label(self, text):
        return [self.letters.index(char) for char in text]

    def pad_sequence(self, seq, value=0):
        return seq + [value] * (self.max_len - len(seq))
