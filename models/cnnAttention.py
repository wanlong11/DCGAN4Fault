import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvAttentionModel, self).__init__()
        self.cnn = BasicCNN(num_classes)
        self.conv_att = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 基本CNN模型的前向传递
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), cnn_out.size(1), 1, 1)

        # 计算Attention权重
        att_weights = F.softmax(self.conv_att(self.cnn.conv2.weight), dim=1)
        att_weights = att_weights.view(att_weights.size(0), att_weights.size(1), 1, 1)

        # 计算Attention特征
        att_feat = (att_weights * cnn_out).sum(dim=1, keepdim=True)

        # 将Attention特征和CNN特征拼接起来
        out = torch.cat([cnn_out, att_feat], dim=1)

        # 用池化层进行降维
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)

        # 全连接层输出
        out = self.cnn.fc(out)
        return out
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x