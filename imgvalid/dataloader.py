import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

classes = ['good', 'bad']
# Define the dataset
class TrainDataset(Dataset):
    def __init__(self, root_dir = '/data/users/zhangyixiang/data/2023/dataset/validator/data'):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((500, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])
        self.file_list = []
        self.label_list = []
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.png'):
                    self.file_list.append(os.path.join(class_path, file_name))
                    self.label_list.append(class_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return img, label

class ImageCaptureDataset(Dataset):
    def __init__(self, file_list):
        self.transform = transforms.Compose([
            transforms.Resize((500, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img