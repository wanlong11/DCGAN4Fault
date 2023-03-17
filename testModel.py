import argparse

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.resNet import ResNet50,ResNet18
from models.resNetWithNLAttention import ResNet50NonLocal,ResNet18NonLocal
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/SDP/data_1/", help="where the all train data is")
parser.add_argument("--checkpoint", type=str,default="", help="checkpoint path")
parser.add_argument("--model",type=str,help="model type")
parser.add_argument("--batch_size",type=int,default=64,help="project name")
parser.add_argument("--num_class",type=int,default=5,help="num of class")
parser.add_argument("--test_data_dir",type=str,help="test data path")

opt = parser.parse_args()

#
#这里每次添加模型都要在这个字典中注册一下
#
modelDict={}
modelDict[ResNet18.__name__]=ResNet18
modelDict[ResNet50.__name__]=ResNet50
modelDict[ResNet50NonLocal.__name__]=ResNet50NonLocal
modelDict[ResNet18NonLocal.__name__]=ResNet18NonLocal



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



# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(*calcuMeanAndStd(path=data_dir))
# ])
# 这里测试集应该使用
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*calcuMeanAndStd(path=opt.data_dir))
])

dataset = ImageFolder(opt.test_data_dir, transform=test_transforms)
# Split the dataset into training and testing sets

test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

model = modelDict[opt.model](num_classes=opt.num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 重载模型模块
if opt.checkpoint != "":
    print("加载模型....")
    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"准确率为{100*correct/total}%")


