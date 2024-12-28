from pyspark import SparkContext
sc =SparkContext.getOrCreate()

from pyspark.ml.image import ImageSchema
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader  # private API
import os
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from functools import reduce
from collections import namedtuple
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
Params = namedtuple('Params', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'momentum', 'seed', 'cuda', 'log_interval'])
args = Params(batch_size=64, test_batch_size=64, epochs=2, lr=0.001, momentum=0.5, seed=1, cuda=use_cuda, log_interval=1)
torch.manual_seed(args.seed)

brain_mri_path = './kaggle_3m'  # Update this path to your local dataset directory
data_mask_path = os.path.join(brain_mri_path, 'data.csv')

class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.images = []
        self.labels = []
        for folder, label in paths:
            if os.path.isdir(folder):
                print(f"Processing folder: {folder}")
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) and filename.endswith('.tif'):
                        self.images.append(file_path)
                        self.labels.append(label)
                        print(f"Added image: {file_path}")
            else:
                print(f"Folder not found: {folder}")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure the image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the CSV file into a DataFrame
df = pd.read_csv(data_mask_path)
# Update column names here
list_files = [(os.path.join(brain_mri_path, df.iloc[i].Patient), int(df.iloc[i].death01) if pd.notnull(df.iloc[i].death01) else 0) for i in range(len(df))]

# Debugging: Print the paths being processed
print("List of files being processed:")
for path in list_files:
    print(path)

from random import shuffle
shuffle(list_files)
Train_size = int(len(list_files) * 0.85)
list_files_train = list_files[:Train_size]
list_files_test = list_files[Train_size:]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.08, 0.08, 0.08], std=[0.12, 0.12, 0.12])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.08, 0.08, 0.08], std=[0.12, 0.12, 0.12])
])

train_loader = torch.utils.data.DataLoader(ImageDataset(list_files_train, train_transform),
                                           batch_size=args.batch_size, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(ImageDataset(list_files_test, test_transform),
                                          batch_size=args.test_batch_size, shuffle=False, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = Net().to(device)

for p in model.parameters():
    p.requires_grad = True

cross_loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = cross_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), train_loss / (batch_idx + 1)))

def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += cross_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

# Training loop
for epoch in range(1, args.epochs + 1):
    train_epoch(epoch, args, model, train_loader, optimizer)
    test_epoch(model, test_loader)
