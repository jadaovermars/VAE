from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
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
        self.fc2 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(2, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z), z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Only reconstruction loss
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar -git mu.pow(2) - logvar.exp())

    return BCE #+ KLD

train_losses = []
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    
    average_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(average_train_loss)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, average_train_loss))


def test(epoch):
    model.eval()
    test_loss = 0
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z = model(data)
            latent_vectors.append(z.cpu().numpy())
            labels.extend(target.numpy())
            test_loss += loss_function(recon_batch, data).item() 
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    if epoch == args.epochs:
        torch.save(model.state_dict(), "model.pt")

        # latent space distributions figure
        cmap = ListedColormap(['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan'])

        latent_vectors = np.concatenate(latent_vectors, axis=0)
        x = latent_vectors[:, 0]
        y = latent_vectors[:, 1]

        # scatter plot of latent space
        plt.figure(figsize=(6, 7.4))
        gs = GridSpec(5, 3, width_ratios=[4.2, 0.2, 1.6], height_ratios=[1.6, 0.2, 4.2, 1, 0.4], wspace=0, hspace=0)
        sctrgrid = plt.subplot(gs[2, 0])
        sctr = sctrgrid.scatter(x, y, c=labels, cmap=cmap, s=10) # colours
        # sctr = sctrgrid.plot(x, y,'.', markersize=1, color=(0.1, 0.1, 0.1, 0.5)) # gaussian distribution
        sctrgrid.set_xlabel('Dimension 1')
        sctrgrid.set_ylabel('Dimension 2')

        # colour legend
        cbargrid = plt.subplot(gs[4, 0])
        plt.colorbar(sctr, cax=cbargrid, orientation='horizontal')
        
        # distribution curves for dimension 1
        xdistr = plt.subplot(gs[0, 0], sharex=sctrgrid)
        xdensity = gaussian_kde(x)
        xs = np.linspace(np.min(x), np.max(x), 100)
        xdistr.plot(xs, xdensity(xs), color='blue')
        xdistr.set_ylabel('Density')
        xdistr.autoscale(enable=True, axis='y')
        xdistr.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # gaussian curve for dimension 1
        mean_x, std_x = norm.fit(x)
        xdistr.plot(xs, norm.pdf(xs, mean_x, std_x), 'r--', label=f'Mean: {mean_x:.2f}\nStd: {std_x:.2f}')
        xdistr.legend(loc='upper right')

        # distribution curves for dimension 2
        ydistr = plt.subplot(gs[2, 2], sharey=sctrgrid)
        ydensity = gaussian_kde(y)
        ys = np.linspace(np.min(y), np.max(y), 100)
        ydistr.plot(ydensity(ys), ys, color='blue')
        ydistr.set_xlabel('Density')
        ydistr.autoscale(enable=True, axis='x')
        ydistr.tick_params(axis='y', which='both', left=False, labelleft=False)

        # gaussian curve for dimension 2
        mean_y, std_y = norm.fit(y)
        ydistr.plot(norm.pdf(ys, mean_y, std_y), ys, 'r--', label=f'Mean: {mean_y:.2f}\nStd: {std_y:.2f}')
        ydistr.legend(loc='lower right')

        plt.suptitle('Latent Space Distributions', fontsize=14, fontweight='bold', y=0.95, va='center')
        plt.show()

if __name__ == "__main__":
    load_model = input("Do you want to load the model? (y/n): ")
    if load_model.lower() == "y":
        load_model = True
    else: 
        load_model = False
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
    if not load_model:
        # training loss graph
        plt.plot(range(1, args.epochs + 1), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Variational Autoencoder Training Loss')
        plt.show()
        

# 8x8 samples from 2d latent space
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
plt.xlabel("Z D1")
plt.ylabel("Z D2")
plt.imshow(figure, cmap='Greys_r')
plt.savefig('fig.jpg')
plt.show()