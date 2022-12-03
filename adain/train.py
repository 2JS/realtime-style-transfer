import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from test import style_transfer

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-4)
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=4000)
parser.add_argument('--test_content', type=str, default='./test2017')
parser.add_argument('--test_style', type=str, default='./test')
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()
tc_tf = test_transform()
ts_tf = test_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
content_test = FlatFolderDataset(args.test_content, tc_tf)
style_test = FlatFolderDataset(args.test_style, ts_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
tc_iter = iter(data.DataLoader(
    content_test, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_test),
    num_workers=args.n_threads
))
ts_iter = iter(data.DataLoader(
    style_test, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_test),
    num_workers=args.n_threads
))

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s, g_t = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('train_content', loss_c.item(), i + 1)
    writer.add_scalar('train_style', loss_s.item(), i + 1)
    print('Epoch {i} || train loss = {loss:.4f}')
    net.eval()
    with torch.no_grad():
        content_test = next(tc_iter).to(device)
        style_test = next(ts_iter).to(device)
        loss_c_valid, loss_s_valid, g_t_valid = network(content_test, style_test)
        loss_c_valid = args.content_weight * loss_c_valid
        loss_s_valid = args.style_weight * loss_s_valid
        loss_valid = loss_c_valid + loss_s_valid

        writer.add_scalar('valid_content', loss_c_valid.item(), i+1)
        writer.add_scalar('valid_style', loss_s_valid.item(), i+1)
        print('Epoch {i} || valid loss = {loss_valid:.4f}')
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict_d = net.decoder.state_dict()
        state_dict_e = net.vgg.state_dict()
        for key in state_dict_d.keys():
            state_dict_d[key] = state_dict_d[key].to(torch.device('cpu'))
        for key in state_dict_e.keys():
            state_dict_e[key] = state_dict_d[key].to(torch.device('cpu'))
        torch.save(g_t, save_dir / 'iter_{:d}.jpg'.format(i + 1))
        torch.save(g_t_valid, save_dir / 'iter_{:d}.jpg'.format(i + 1))
        torch.save(state_dict_d, save_dir /
                   'decoder_iter_{:d}.pth'.format(i + 1))
        torch.save(state_dict_e, save_dir /
                   'encoder_iter_{:d}.pth'.format(i + 1))
writer.close()
