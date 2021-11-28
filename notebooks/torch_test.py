#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.net import Net
from src.models.onboard_dataset import OnboardDataset


# In[30]:


class TrainModel:
    def __init__(self):
        torch.manual_seed(1)
        self.train_set = None
        self.valid_set = None
        self.device = torch.device("cpu")
        self.log_interval = 100
        self.model = Net().to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_batch_size = 64
        self.valid_batch_size = 1000
        self.transform = None
        self.train_loader = None
        self.valid_loader = None
        self._set_transform()

    def _set_data_loader(self):
        self.train_loader = DataLoader(OnboardDataset(path=self.train_set, transform=self.transform),
                                       batch_size=self.train_batch_size, shuffle=True)
        self.valid_loader = DataLoader(OnboardDataset(path=self.valid_set, transform=self.transform),
                                       batch_size=self.valid_batch_size, shuffle=True)

    def _set_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _train_model(self, epoch, lr):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                
    def _valid_model(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.valid_loader.dataset)

        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
              (val_loss, correct, len(self.valid_loader.dataset),
               100. * correct / len(self.valid_loader.dataset)))
        
    def load_model(self, model_path):
        self.model = torch.load(model_path)

    def train(self, train_set, valid_set, epochs, lr):
        self.train_set = train_set
        self.valid_set = valid_set
        self._set_data_loader()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            self._train_model(epoch, lr)
            self._valid_model()

    def save_model(self, save_model_path):
        torch.save(self.model, os.path.join(save_model_path, "model.pt"))


# In[31]:


if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train(epochs=2, lr=0.01,
                      train_set="./data/processed/train_set.npz",
                      valid_set="./data/processed/valid_set.npz")
    train_model.save_model(save_model_path="./models/")

