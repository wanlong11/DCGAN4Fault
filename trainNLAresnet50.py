import argparse

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.resNetWithNLAttention import ResNet50NonLocal
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/SDP/data_1/", help="where the data is")
parser.add_argument("--SB_before", type=int, default=500, help="number of save best model epochs begin ")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--eval_interval", type=int, default=1, help="number of evaluation interval step")
parser.add_argument("--checkpoint", type=str,default="", help="checkpoint path")
parser.add_argument("--tensorboard_dir", type=str ,help="tensorboard dirationary")
parser.add_argument("--save_dir",type=str,help="the path of save checkpoint")
parser.add_argument("--project_name",type=str,help="project name")
parser.add_argument("--batch_size",type=int,default=64,help="project name")
parser.add_argument("--test_dir",type=str,help="project name")
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

writer = SummaryWriter(log_dir=opt.tensorboard_dir)

data_dir = opt.data_dir
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*calcuMeanAndStd(path=data_dir))
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*calcuMeanAndStd(path=data_dir))
])

dataset = ImageFolder(data_dir, transform=train_transforms)
test_dataset = ImageFolder(opt.test_dir,transform=test_transforms)
# Split the dataset into training and testing sets
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])

# Define the data loaders for training and testing sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


model = ResNet50NonLocal(num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
start_epoch = 0

# 重载模型模块
if opt.checkpoint != "":
    print("重新加载模型....")
    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

# Train the model
os.makedirs(opt.save_dir,exist_ok=True)
global_step=0
best_acc=0
for epoch in range(start_epoch, opt.n_epochs):
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss, global_step=global_step)
        global_step+=1
        print(loss.item())
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
            writer.add_scalar('test/Loss', loss, global_step=global_step)
        writer.add_scalar('test/Accuracy', 100 * correct / total, global_step=global_step)
    # evaluate the model
    if epoch > opt.SB_before and epoch % opt.eval_interval == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images=images.to(device)
                labels=labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                writer.add_scalar('Eval/Loss', loss, global_step=global_step)
            writer.add_scalar('Eval/Accuracy', 100*correct / total, global_step=global_step)
            #提前停止策略
            if best_acc<(correct/total):
                best_acc=(correct/total)
                try:
                    temp = os.listdir(opt.save_dir)
                    for tName in temp:
                        if 'best' in tName:
                            os.remove('./' + opt.save_dir + '/' + tName)
                            print('删除模型成功')
                except:
                    print("删除失败")
                torch.save(model, opt.save_dir + '/' + opt.project_name + 'best.pt')


torch.save(model, opt.save_dir + '/' + opt.project_name + 'last.pt')
