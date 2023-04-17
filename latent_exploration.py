
import math
import random
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pathlib
import itertools

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import numpy as np

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)

import pydiffvg


latent_dim = 64
# batch_size = 256
batch_size = 61
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
# n_epochs = 50
n_epochs = -2
# kl_weight = 0.5
kl_weight = 1. / batch_size
img_size = 64
conditional = True
# num_letters = 36
num_letters = 26

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def indxofletter(letter):
    return ord(letter) - 65

def idnxtoletter(idx):
    return chr(idx + 65)

def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


class CustomDataset(Dataset):
    def __init__(self, images_path, transform):
        self.transform = transform
        self.images_path = images_path

        self.paths = list(pathlib.Path(images_path).glob("*/*.png"))
        self.classes = [indxofletter(p.parents[0].name) for p in self.paths] #p.parents[0].name ->letra #p.parents[0] -> diretoria da pasta anterior (outputs/Letra)
        # print(self.paths)
        # print(self.classes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]

        clss = self.classes[idx]

        image = Image.open(image_path)

        image = self.transform(image)

        return image, clss, image_path.stem


images_path = 'outputs/'

dataset = CustomDataset(images_path, transform=transform, )
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,
                  canvas_height,
                  samples,
                  samples,
                  0,
                  None,
                  *scene_args)
    return img


def opacityStroke2diffvg(strokes, canvas_size=128, debug=False, relative=True, force_cpu=True):
    dev = strokes.device
    if force_cpu:
        strokes = strokes.to("cpu")

    """Rasterize strokes given in (dx, dy, opacity) sequence format."""
    bs, nsegs, dims = strokes.shape
    out = []

    start = time.time()
    for batch_idx, stroke in enumerate(strokes):

        if relative:  # Absolute coordinates
            all_points = stroke[..., :2].cumsum(0)
        else:
            all_points = stroke[..., :2]

        # all_opacities = stroke[..., 2]

        # Transform from [-1, 1] to canvas coordinates
        # Make sure points are in canvas
        all_points = 0.5 * (all_points + 1.0) * canvas_size
        # all_points = th.clamp(0.5*(all_points + 1.0), 0, 1) * canvas_size

        # Avoid overlapping points
        eps = 1e-4
        all_points = all_points + eps * torch.randn_like(all_points)

        shapes = []
        shape_groups = []

        for start_idx in range(0, nsegs-1):
            points = all_points[start_idx:start_idx+2].contiguous().float()

            num_ctrl_pts = torch.zeros(points.shape[0] - 1, dtype=torch.int32)
            # width = torch.ones(1)
            width = torch.tensor([2.])

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)

            shapes.append(path)

            color = torch.ones(4, device=dev)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)

        # Rasterize only if there are shapes
        if shapes:
            inner_start = time.time()
            out.append(render(canvas_size, canvas_size, shapes, shape_groups,
                              samples=4))
            if debug:
                inner_elapsed = time.time() - inner_start
                print("diffvg call took %.2fms" % inner_elapsed)
        else:
            out.append(torch.zeros(canvas_size, canvas_size, 4, device=strokes.device))

    if debug:
        elapsed = (time.time() - start)*1000
        print("rendering took %.2fms" % elapsed)

    images = torch.stack(out, 0).permute(0, 3, 1, 2).contiguous().mean(1, keepdim=True)

    # Return data on the same device as input
    return images.to(dev)


def interpolate(model, sketch_decoder, epoch):
    interpolation_step = 10
    interpolation_size = 1.0 / interpolation_step

    dataloader_2 = DataLoader(dataset, batch_size=4, shuffle=True)

    images = []
    for letter in range(26):
        print(letter)
        k = int((2700 * letter) / 4)
        samples, labels, names = next(itertools.islice(dataloader_2, k, None)) # samples (4, 3, 64, 64)
        samples = samples.view(2, 2, *samples.shape[1:]) # samples (2, 2, 3, 64, 64)
        labels = labels.view(2, 2, *labels.shape[1:]) # labels: que letra é (2, 2, 26)
        for i in range(samples.shape[0]):
            sample = samples[i].to(device)
            label = labels[i].to(device)

            y, z, mu, log_var = model(sample, label)

            new_z = []
            inter = 0.0
            while inter < 1.0:
                new_z.append(torch.lerp(z[0], z[1], inter))
                inter += interpolation_size

            new_z = torch.stack(new_z, dim=0)
            letters = sketch_decoder(new_z)

            for j in range(interpolation_step):
                writeFile(letters[j], f"samples_{names[i*2]}_{labels[i][0]}__{names[(i*2)+1]}_{labels[i][1]}/", f"file_{j}.txt", names[i*2], names[(i*2)+1], idnxtoletter(labels[i][0]), idnxtoletter(labels[i][1]))

            # print(letters.shape)
            sketch_img = opacityStroke2diffvg(letters, canvas_size=img_size, debug=False, force_cpu=False, relative=False)
            images.extend([sample[0]])
            images.extend([*sketch_img])
            images.extend([sample[1]])

    images = torch.stack(images, dim=0)
    save_image(images, "images/samples_{}.png".format(epoch), nrow=13)


def writeFile(strokes, folder, name, font1, font2, letter1, letter2):
    # cretae fodler
    if not os.path.exists("txt_output/" + folder):
        os.makedirs("txt_output/" + folder)

    f = open("txt_output/" + folder + name, "w")
    f.write("{},{}\n".format(font1, letter1))
    f.write("{},{}\n".format(font2, letter2))

    for i in range(len(strokes)):
        f.write(f"{strokes[i, 0].item()},{strokes[i, 1].item()}\n")
    f.close()

def _onehot(label):
    bs = label.shape[0]
    label_onehot = label.new(bs, num_letters)
    label_onehot = label_onehot.zero_()
    label_onehot.scatter_(1, label.unsqueeze(1), 1)
    return label_onehot.float().to(device)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=latent_dim):
        super(VAE, self).__init__()

        self.conditional = conditional

        ncond = 0
        if self.conditional:  # one hot encoded input for conditional model
            ncond = num_letters

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + ncond, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            UnFlatten(size=z_dim + ncond),
            nn.ConvTranspose2d(z_dim + ncond, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, label):
        if self.conditional:
            label_onehot = _onehot(label)
            label_onehot = label_onehot.view(x.shape[0], num_letters, 1, 1).repeat(1, 1, img_size, img_size)
            x = self.encoder(torch.cat([x, label_onehot], 1))
        else:
            x = self.encoder(x)

        z, mu, logvar = self.bottleneck(x)

        return z, mu, logvar

    def decode(self, z, label):
        if self.conditional:
            label_onehot = _onehot(label)
            z = torch.cat([z, label_onehot], 1)

        y = self.decoder(z)
        return y

    def forward(self, x, label):
        z, mu, logvar = self.encode(x, label)

        y = self.decode(z, label)

        return y, z, mu, logvar


class SketchDecoder(nn.Module):
    """
    The decoder outputs a sequence where each time step models (dx, dy,
    opacity).
    """

    def __init__(self, sequence_length=30, hidden_size=512, dropout=0.9,
                 zdim=latent_dim, num_layers=3):
        super(SketchDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.zdim = zdim

        # Maps the latent vector to an initial cell/hidden vector
        self.hidden_cell_predictor = nn.Linear(zdim, 2 * hidden_size * num_layers)

        self.lstm = nn.LSTM(
            zdim, hidden_size,
            num_layers=self.num_layers, dropout=dropout,
            batch_first=True)

        self.dxdy_predictor = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Tanh(),
        )

    def forward(self, z, hidden_and_cell=None):
        # Every step in the sequence takes the latent vector as input so we
        # replicate it here
        bs = z.shape[0]
        steps = self.sequence_length  # no need to predict the start of sequence
        expanded_z = z.unsqueeze(1).repeat(1, steps, 1)

        if hidden_and_cell is None:
            # Initialize from latent vector
            hidden_and_cell = self.hidden_cell_predictor(torch.tanh(z))
            hidden = hidden_and_cell[:, :self.hidden_size * self.num_layers]
            hidden = hidden.view(-1, self.num_layers, self.hidden_size)
            hidden = hidden.permute(1, 0, 2).contiguous()
            # hidden = hidden.unsqueeze(1).contiguous()
            cell = hidden_and_cell[:, self.hidden_size * self.num_layers:]
            cell = cell.view(-1, self.num_layers, self.hidden_size)
            cell = cell.permute(1, 0, 2).contiguous()
            # cell = cell.unsqueeze(1).contiguous()
            hidden_and_cell = (hidden, cell)

        outputs, hidden_and_cell = self.lstm(expanded_z, hidden_and_cell)
        hidden, cell = hidden_and_cell

        dxdy = self.dxdy_predictor(
            outputs.reshape(bs * steps, self.hidden_size)).view(bs, steps, -1)

        strokes = dxdy

        return strokes


pydiffvg.set_use_gpu(torch.cuda.is_available())

model = VAE(image_channels=1).to(device)

model.load_state_dict(torch.load("models/encoder.pth"))
model.eval()


for letter in tqdm(range(26)):
    print(letter)
    k_start = int((2623 * letter) / batch_size)
    k_end = int((2623 * (letter + 1)) / batch_size)

    z_values = []
    z_names = []
    z_labels = []
    with torch.no_grad():
        for x, labels, names in itertools.islice(dataloader, k_start, k_end):
            x = x.to(device)
            labels = labels.to(device)

            y, z, mu, log_var = model(x, labels)

            z_values.extend(z)
            z_names.extend(names)
            z_labels.extend(labels)

    z_values = torch.stack(z_values, dim=0).cpu().detach().numpy()

    print(z_values.shape)
    print(len(z_names))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(z_values)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    img_size_output = 128

    # creating new Image object
    img = Image.new("RGB", (50 * img_size_output, 50 * img_size_output))

    min_x = min(tsne_results[:, 0])
    max_x = max(tsne_results[:, 0])
    min_y = min(tsne_results[:, 1])
    max_y = max(tsne_results[:, 1])

    print(min(tsne_results[:, 0]), max(tsne_results[:, 0]), min(tsne_results[:, 1]), max(tsne_results[:, 1]))

    for x in tqdm(range(50)):
        for y in range(50):
            min_dist = 9999.0
            index = 0

            for t, tsne_result in enumerate(tsne_results):
                x_pos = map_number(tsne_result[0], min_x, max_x, 0, 50)
                y_pos = map_number(tsne_result[1], min_y, max_y, 0, 50)

                dist = math.sqrt(math.pow(x_pos - x, 2) + math.pow(y_pos - y, 2))

                if min_dist > dist:
                    min_dist = dist
                    index = t

            if min_dist < 1.0:
                l = idnxtoletter(z_labels[index])
                image_path = images_path + l + "/" + z_names[index] + ".png"
                letter_img = Image.open(image_path)
                letter_img = letter_img.resize((img_size_output, img_size_output))

                img.paste(letter_img, (x * img_size_output, y * img_size_output))

    img.save(f'latent_space_{letter}.png')
