import numpy as np
from torch.utils.data import Dataset


class OnboardDataset(Dataset):
    def __init__(self, path, transform=None):
        dataset = np.load(path)
        self.x = dataset['x']
        self.y = dataset['y']
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = int(self.y[item])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
