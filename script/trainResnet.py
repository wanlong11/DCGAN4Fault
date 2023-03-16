import argparse

import torch.nn as nn

from model.resNet import ResNet18

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/SDP/data_1/", help="where the data is")
parser.add_argument("--SB_before", type=int, default=500, help="number of save best model epochs begin ")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--eval_interval", type=int, default=1, help="number of evaluation interval step")
opt = parser.parse_args()


def calcuMeanAndStd(path):
    # 计算路径下所有图像的通道mean和std
    # Define the dataset and data loader
    dataset = ImageFolder(path, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Define variables to store the sum and square sum of pixel values
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)

    # Loop over the data and accumulate the sum and square sum
    for i, (inputs, _) in enumerate(dataloader):
        sum_ += torch.mean(inputs, dim=[0, 2, 3])
        sum_sq += torch.mean(inputs ** 2, dim=[0, 2, 3])

    # Compute the mean and standard deviation
    mean = sum_ / len(dataset)
    std = torch.sqrt(sum_sq / len(dataset) - mean ** 2)

    return list(mean), list(std)  # 这里有可能会有问题 check


data_dir = opt.data_dir
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*calcuMeanAndStd(path=data_dir))
])
# 这里测试集应该使用
# test_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(*calcuMeanAndStd(path=data_dir))
# ])

dataset = ImageFolder(data_dir, transform=train_transforms)

# Split the dataset into training and testing sets
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])

# Define the data loaders for training and testing sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

model = ResNet18(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(opt.n_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())

    # evaluate the model
    if epoch > opt.SB_before and epoch % opt.eval_interval == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))
