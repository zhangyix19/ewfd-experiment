import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import SimpleCNN
from dataloader import TrainDataset
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs')
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
num_epochs = args.epochs

# Load the dataset and split into train and validation sets
dataset_path = '/data/users/zhangyixiang/data/2023/dataset/validator/data'
logging.info("Loading dataset from {}".format(dataset_path))
dataset = CustomDataset(dataset_path)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
logging.info("Train dataset size: {}".format(len(train_dataset)))
logging.info("Validation dataset size: {}".format(len(valid_dataset)))


# Define the data loaders
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=32)
validloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=32)


# Train the model
model = SimpleCNN().to(device)
model.train()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    logging.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
    running_loss = 0.0
    true_count = 0
    sum_count = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        true_count += (outputs.argmax(dim=1) == labels).sum().item()
        sum_count += labels.size(0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            logging.info('Train: [%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 10, true_count / sum_count))
            running_loss = 0.0
            true_count = 0
            sum_count = 0

    # Evaluate the model on the validation set
    
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logging.info("Valid: [%d] accuracy: %.3f" % (epoch + 1, correct / total))
# save the model
model_path = '/data/users/zhangyixiang/data/2023/dataset/validator/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
logging.info("Saving model to {}".format(model_path))
torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))

