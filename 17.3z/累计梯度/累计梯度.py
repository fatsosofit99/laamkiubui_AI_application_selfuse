import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Classifier(nn.Module):
    #TODO
    def __init__(self,input_dim,num_classes):
        super(Classifier,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.net(x)


def load_data(npz_path: str) -> TensorDataset:
    data = np.load(npz_path)
    X = torch.from_numpy(data['X'])
    y = torch.from_numpy(data['y'])
    return TensorDataset(X, y)

def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, accumulation_steps: int, device: torch.device) -> None:
    #TODO
    model.train()
    model.to(device)
    optimizer.zero_grad()
    for i,(inputs,labels) in enumerate(dataloader):
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss = loss/accumulation_steps
        loss.backward()

        if(i+1)%accumulation_steps ==0 or (i+1)==len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

if __name__ == '__main__':
    set_seed()
    npz_path = 'classification_data.npz'
    dataset = load_data(npz_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_dim = dataset.tensors[0].shape[1]
    num_classes = len(torch.unique(dataset.tensors[1]))

    model = Classifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accumulation_steps = 4
    for epoch in range(30):
        train(model, dataloader, optimizer, criterion, accumulation_steps, device)