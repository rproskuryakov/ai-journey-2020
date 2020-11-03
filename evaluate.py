import os
import pathlib

from PIL import Image
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as tf

from src.models.resnet_model import ResNet50Network
from src.models.baseline_model import BaselineNetwork
from src.decoders import CTCDecoder


if __name__ == "__main__":
    MODEL_PATH = pathlib.Path("models/checkpoint_torch/v3_resnet_50/best_model_epoch_58.pth")
    INPUT_PATH = pathlib.Path("/data")
    OUTPUT_PATH = pathlib.Path("/output")

    LETTERS = ['_', ' ', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 'a', 'b', 'c', 'd',
               'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', '|', '×', 'ǂ', 'а', 'б', 'в',
               'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч',
               'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'і', 'ѣ', '–', '…', '⊕', '⊗']

    # Define transforms
    transformer = tf.Compose([
        tf.Resize((128, 1024)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    # transformer = BaselineTransformer()

    # Load model
    model = ResNet50Network(len(LETTERS), pretrained_resnet=False)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Decoder
    decoder = CTCDecoder(letters=LETTERS)

    # List all images filenames
    images = os.listdir(INPUT_PATH)
    for image_name in images:
        filename = pathlib.Path(image_name)
        image = Image.open(INPUT_PATH / filename)
        image = transformer(image).unsqueeze(0)
        output = model(image)
        out_val, out_lens = decoder(output)
        pred_texts, _ = decoder.decode(out_val, ctc_lens=out_lens)

        with open(OUTPUT_PATH / filename.with_suffix(".txt"), "w", encoding="utf8") as file:
            file.write(pred_texts[0].strip())
