import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        Q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                       self.d_model)

        output = self.Wo(context)

        return output


class ImageMultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageMultiClassifier, self).__init__()
        self.ln = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.attention = MultiHeadAttention(d_model=64, num_heads=4)

        self.fc1 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.attention(x)

        out += self.ln(out)  # 残差链接

        out = self.conv1(x)  # 卷1
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)  # 池1

        out = self.conv2(out)  # 卷2
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)  # 池2

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)

        return out
# 包含三个卷积层、两个残差块和一个多头注意力机制的多分类模型。其中，多头注意力机制用于捕捉图像中不同部分之间的关系，残差块用于加快模型的收敛速度，批归一化层用于加快训练速度，而卷积层用于提取图像的特征。
