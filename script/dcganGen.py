import torch
import numpy as np
from torchvision.utils import save_image
from model.dcgan import Generator
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="GenImage", help="path of output")
parser.add_argument("--batch_size", type=int, default="1", help="gen Batch Size")
parser.add_argument("--model_file", type=str, default="1.pt", help="the parameter file")
parser.add_argument("--latent_dim", type=int, default=256, help="the parameter file")
parser.add_argument("--gen_num", type=int, default=100, help="number of Generate image batch")
opt = parser.parse_args()
print(opt)
model = Generator()
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model.load_state_dict(torch.load(opt.model_file))

# generate
for num in range(opt.gen_num):
    seed = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    out_img = model(seed)

    save_image(out_img[:opt.batch_size - 1] + "images/%d.png" % num, nrow=5)
