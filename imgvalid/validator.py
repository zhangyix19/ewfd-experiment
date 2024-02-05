import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import SimpleCNN
from dataloader import ImageCaptureDataset, classes
from PIL import Image
import argparse
from tqdm import tqdm


class Validator:
    def __init__(
        self,
        device,
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.pth"),
    ):
        self.device = device
        assert os.path.exists(model_path)
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def validate(self, img_file_list, batch_size=128):
        dataset = ImageCaptureDataset(img_file_list)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=32
        )
        valid_list = []
        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, img in enumerate(dataloader):
                    pbar.set_description(desc="Validating images")
                    pbar.set_postfix(
                        current=len(valid_list), valid=sum([c == "good" for c in valid_list])
                    )
                    pbar.update(1)
                    img = img.to(self.device)
                    output = self.model(img)
                    output = torch.argmax(output, dim=1)
                    valid_list += [classes[i] for i in output]

        valid_list = [c == "good" for c in valid_list]
        print("Number of valid images / total images:", sum(valid_list), "/", len(valid_list))
        return valid_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--imgdir", type=str, default=None)
    args = parser.parse_args()
    validator = Validator("cpu")
    if args.imgdir is not None:
        img_file_list = [os.path.join(args.imgdir, f) for f in os.listdir(args.imgdir)]
        valid_list = validator.validate(img_file_list)
        # count the number of each class
        print("The number of good images is", sum(valid_list))
        print("The number of bad images is", len(valid_list) - sum(valid_list))
    if args.img is not None:
        img_file_list = [args.img]
        valid_list = validator.validate(img_file_list)
        print("The image is", classes[valid_list[0]])
