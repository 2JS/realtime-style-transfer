import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper
from function import adaptive_instance_normalization, coral

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
    transform_list = []
    transform_list.append(transforms.Resize(512))
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


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

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-3)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=500)
parser.add_argument('--test_content', type=str, default='./test2017')
parser.add_argument('--test_style', type=str, default='./test')
parser.add_argument('--output_dir', default='./outputs')
parser.add_argument('--content', default='./input/content')
parser.add_argument('--style', default='./input/style')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
args = parser.parse_args()

device = torch.device('cuda')
do_interpolation = False

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

content_dir = Path(args.content)
content_paths = [f for f in content_dir.glob('*')]
style_dir = Path(args.style)
style_paths = [f for f in style_dir.glob('*')]

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

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    network.train()
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
    network.eval()
    for content_path in content_paths:
      for style_path in style_paths:
          content = tc_tf(Image.open(str(content_path)))
          style = ts_tf(Image.open(str(style_path)))
          style = style.to(device).unsqueeze(0)
          content = content.to(device).unsqueeze(0)
          with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                      1)
          output = output.cpu()

          output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
              content_path.stem, style_path.stem, args.save_ext)
          save_image(output, str(output_name))
    if (i) % args.save_model_interval == 0 or (i) == args.max_iter:
        print(f'Epoch {i} || train loss = {loss:.4f}')
        state_dict_d = net.decoder.state_dict()
        state_dict_e = net.vgg.state_dict()
        for key in state_dict_d.keys():
            state_dict_d[key] = state_dict_d[key].to(torch.device('cpu'))
        for key in state_dict_e.keys():
            state_dict_e[key] = state_dict_e[key].to(torch.device('cpu'))
        torch.save(g_t, save_dir / 'iter_{:d}.jpg'.format(i))
        torch.save(state_dict_d, save_dir /
                   'decoder_iter_{:d}.pth'.format(i))
        torch.save(state_dict_e, save_dir /
                   'encoder_iter_{:d}.pth'.format(i))
writer.close()
