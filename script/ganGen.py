import os

import torch
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import argparse
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="ganGenImage", help="path of output")
parser.add_argument("--batch_size", type=int, default=1, help="gen Batch Size")
parser.add_argument("--model_file", type=str, default="1.pt", help="the parameter file")
parser.add_argument("--latent_dim", type=int, default=256, help="the parameter file")
parser.add_argument("--gen_num", type=int, default=100, help="number of Generate image batch")
parser.add_argument("--img_size", type=int, default=256, help="the size of generate images")
parser.add_argument("--channels", type=int, default=3, help="image channel")

opt = parser.parse_args()
print(opt)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = torch.load(opt.model_file)

os.makedirs(opt.out_dir, exist_ok=True)
# generate
for num in range(opt.gen_num):
    seed = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    out_img = model(seed)

    save_image(out_img[:], f"{opt.out_dir}/%d.png" % num, normalize=True)
    print(f"complete{num / opt.gen_num}")
