import torch
import torchvision
import torchvision.transforms as transforms

# Download MNIST and save as NumPy arrays
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

images = torch.stack([dataset[i][0] for i in range(len(dataset))])
labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

# Save to .npz (easy to load with NumPy)
import numpy as np
np.savez("mnist_train.npz", images=images.numpy(), labels=labels.numpy())

