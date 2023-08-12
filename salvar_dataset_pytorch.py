import torch
from torchvision import datasets, transforms

# Definindo as transformações (por exemplo, normalização, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Carregando o ImageFolder
path_to_data = 'databases/CRD'
dataset = datasets.ImageFolder(path_to_data, transform=transform)
data_list = [(data, label) for data, label in dataset]

torch.save(data_list, 'CRD_equilibrado.pt')
