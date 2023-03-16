import argparse
import os
import numpy as np
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
import torch.nn as nn
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

cuda = True if torch.cuda.is_available() else False
os.makedirs(opt.project_name + "images", exist_ok=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

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

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# In[3]:


imgs = glob.glob(opt.img_dir + "*")

species = ['sate']

species_to_idx = dict((c, i) for i, c in enumerate(species))

idx_to_species = dict((v, k) for k, v in species_to_idx.items())

labels = []
for img in imgs:
    for i, c in enumerate(species):
        if c in img:
            labels.append(i)

transforms = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


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


dataset = WT_dataset(imgs, labels)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
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

training_data = pd.DataFrame(columns=['epoch', 'd_loss', 'g_loss'])

os.makedirs(opt.project_name, exist_ok=True)
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

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
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'gbest.pt')
                print("gbest模型保存完毕")
            else:
                try:
                    temp = os.listdir(opt.project_name)
                    for tName in temp:
                        if 'gbest' in tName:
                            os.remove('./' + opt.project_name + '/' + tName)
                            print('删除gbest模型成功')
                except:
                    print('删除gbest模型失败')
                best_gloss = g_loss
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'gbest.pt')
                print("gbest模型保存完毕")
        # Dloss最优
        if best_dloss > d_loss and epoch > opt.SB_before:
            if best_dloss == 1e5:
                best_dloss = d_loss
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'dbest.pt')
                print("dbest模型保存完毕")
            else:
                try:
                    temp = os.listdir(opt.project_name)
                    for tName in temp:
                        if 'dbest' in tName:
                            os.remove('./' + opt.project_name + '/' + tName)
                            print('删除dbest模型成功')
                except:
                    print('删除dbest模型失败')
                best_gloss = g_loss
                torch.save(generator, opt.project_name + '/G3' + str(epoch) + 'dbest.pt')
                print("dbest模型保存完毕")

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:9], opt.project_name + "images/%d.png" % batches_done, nrow=3, normalize=True)

torch.save(generator, opt.project_name + '/G3last.pt')

training_data.to_csv(opt.project_name + "/train_data")
