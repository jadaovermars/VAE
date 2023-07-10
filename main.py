from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt # ***
import numpy as np # ***
from matplotlib.colors import ListedColormap # ***
from scipy.stats import gaussian_kde
from scipy.stats import norm
from matplotlib.gridspec import GridSpec

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, 2)
        self.fc22 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(2, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    latent_vectors = [] # ***
    labels = [] # ***
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            latent_vectors.append(mu.cpu().numpy()) # ***
            labels.extend(target.numpy()) # ***
            test_loss += loss_function(recon_batch, data, mu, logvar).item() 
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    # ***
    if epoch == args.epochs:
        cmap = ListedColormap(['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        torch.save(model.state_dict(), "model.pt")
        x = latent_vectors[:, 0]
        y = latent_vectors[:, 1]

        plt.figure(figsize=(8, 8))
        gs = GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[1, 3, 0.2])
        ax1 = plt.subplot(gs[1, 0])
        scatter = ax1.scatter(x, y, c=labels, cmap=cmap, s=10)

        ax1_legend = plt.subplot(gs[2, 0])  # Occupies all rows, third column
        plt.colorbar(scatter, cax=ax1_legend, orientation='horizontal')
        
        # Distribution curves for dimension 1
        ax2 = plt.subplot(gs[0, 0], sharex=ax1)
        density = gaussian_kde(x)
        xs = np.linspace(np.min(x), np.max(x), 100)
        ax2.plot(xs, density(xs), color='blue')
        ax2.set_ylabel('Density')
        ax2.autoscale(enable=True, axis='y')
        # ax2.set_xticks([])

        # Overlay Gaussian curve for dimension 1
        mean_x, std_x = norm.fit(x)
        ax2.plot(xs, norm.pdf(xs, mean_x, std_x), 'r--', label=f'Gaussian Fit\nMean: {mean_x:.2f}\nStd: {std_x:.2f}')
        ax2.legend()

        # Distribution curves for dimension 2
        ax3 = plt.subplot(gs[1, 1], sharey=ax1)
        density = gaussian_kde(y)
        ys = np.linspace(np.min(y), np.max(y), 100)
        ax3.plot(density(ys), ys, color='blue')
        ax3.set_xlabel('Density')
        ax3.autoscale(enable=True, axis='x')
        # ax3.set_yticks([])


        # Overlay Gaussian curve for dimension 2
        mean_y, std_y = norm.fit(y)
        ax3.plot(norm.pdf(ys, mean_y, std_y), ys, 'r--', label=f'Gaussian Fit\nMean: {mean_y:.2f}\nStd: {std_y:.2f}')
        ax3.legend()

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # TODO: for poster add point to graph that is hollow(black border white inside)
        # the x,y would be the cords for broken point.png 
        # then after we have exported the graph we can put a line pointing to the point and display broken.png

        # plt.xlabel('Latent Dimension 1')
        # plt.ylabel('Latent Dimension 2')
        plt.suptitle('Latent Space Distributions', fontsize=14, fontweight='bold', y=0.95, va='center')
        plt.show()
        # ***

if __name__ == "__main__":
    load_model = input("Do you want to load the model? (y/n):")
    if load_model.lower() == "y":
        load_model = True
    else: 
        load_model = False
    print(load_model)
    if load_model:
        model.load_state_dict(torch.load("model.pt"))
        print("Loading model...")
    for epoch in range(1, args.epochs + 1):
        if not load_model:
            train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
        tensor = torch.tensor([-2.838, 0.01]).to(device)
        tensor = model.decode(tensor).cpu()
        save_image(tensor.view(1, 1, 28, 28),
                    'results/broken.png')

        

# show 2d picture thingy
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]


for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = torch.Tensor([[xi, yi]]).to(device)
        x_decoded = model.decode(z_sample).cpu().detach().numpy()
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = (n - 1) * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 4)
sample_range_y = np.round(grid_y, 4)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("Z [0]")
plt.ylabel("Z [1]")
plt.imshow(figure, cmap='Greys_r')
plt.savefig('fig.jpg')
plt.show()