import argparse
import os
import numpy as np
import math
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default="../data/SDP/data_1/", help="input data from where")
parser.add_argument("--SB_before", type=int, default=500, help="number of save best model epochs begin ")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--project_name", type=str, default='SATEF1', help="project name (model save path)")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
os.makedirs(opt.project_name + "images", exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()





imgs = glob.glob(img_dir)




species = ['sate']





species_to_idx = dict((c, i) for i, c in enumerate(species))



idx_to_species = dict((v, k) for k, v in species_to_idx.items())










labels = []
for img in imgs:
    for i, c in enumerate(species):
        if c in img:
            labels.append(i)




transforms = transforms.Compose([
    transforms.Resize((opt.img_size,opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


# In[13]:


class WT_dataset(data.Dataset):
    def __init__(self, imgs_path, lables):
        self.imgs_path = imgs_path
        self.lables = lables

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        lable = self.lables[index]

        pil_img = Image.open(img_path)
        pil_img = pil_img.convert("RGB")
        pil_img = transforms(pil_img)
        return pil_img, lable

    def __len__(self):
        return len(self.imgs_path)


# In[14]:


dataset = WT_dataset(imgs, labels)




# os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(

    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
import pandas as pd

best_gloss, best_dloss = 1e5, 1e5


for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        training_data.loc[epoch] = {"epoch": epoch, "d_loss": d_loss.item(), "g_loss": g_loss.item()}

        # 存储g_loss最优
        if best_gloss > g_loss and epoch > opt.SB_before:
            if best_gloss == 1e5:
                best_gloss = g_loss
                torch.save(generator, opt.project_name + '/gG3' + str(epoch) + 'ggbest.pt')
                print("ggbest模型保存完毕")
            else:
                try:
                    temp = os.listdir(opt.project_name)
                    for tName in temp:
                        if 'ggbest' in tName:
                            os.remove('./' + opt.project_name + '/' + tName)
                            print('删除ggbest模型成功')
                except:
                    print('删除ggbest模型失败')
                best_gloss = g_loss
                torch.save(generator, opt.project_name + '/gG3' + str(epoch) + 'ggbest.pt')
                print("ggbest模型保存完毕")
        # Dloss最优
        if best_dloss > d_loss and epoch > opt.SB_before:
            if best_dloss == 1e5:
                best_dloss = d_loss
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'gdbest.pt')
                print("gdbest模型保存完毕")
            else:
                try:
                    temp = os.listdir(opt.project_name)
                    for tName in temp:
                        if 'gdbest' in tName:
                            os.remove('./' + opt.project_name + '/' + tName)
                            print('删除gdbest模型成功')
                except:
                    print('删除gdbest模型失败')
                best_gloss = g_loss
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'gdbest.pt')
                print("gdbest模型保存完毕")

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:9], opt.project_name + "gimages/%d.png" % batches_done, nrow=3, normalize=True)

torch.save(generator, opt.project_name + '/ganG3last.pt')

training_data.to_csv(opt.project_name + "/gantrain_data")
