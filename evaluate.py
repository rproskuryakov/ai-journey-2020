import os
import pathlib

from PIL import Image
import torch
import torchvision.transforms as tf

from src.models.resnet_model import ResNetNetwork
from src.decoders import GreedyDecoder

if __name__ == "__main__":
    MODEL_PATH = pathlib.Path("models/checkpoint_torch/v2_resnet/best_model_epoch_78.pth")
    INPUT_PATH = pathlib.Path("data/")
    OUTPUT_PATH = pathlib.Path("output/")
    RESNET_STATE_PATH = pathlib.Path("models/resnet18.pt")
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

    # Load model
    resnet_dict = torch.load(RESNET_STATE_PATH)
    model = ResNetNetwork(len(LETTERS), resnet_state_dict=resnet_dict)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Decoder
    decoder = GreedyDecoder(letters=LETTERS)

    # List add images filenames
    images = os.listdir(INPUT_PATH)
    for image_name in images:
        filename = pathlib.Path(image_name)
        image = Image.open(INPUT_PATH / filename)
        image = transformer(image).unsqueeze(0)
        output = model(image)
        out_val = decoder(output)
        pred_texts, _ = decoder.decode(out_val)
        print(pred_texts)

        with open(OUTPUT_PATH / filename.with_suffix(".txt"), "w") as file:
            file.write(pred_texts[0])
