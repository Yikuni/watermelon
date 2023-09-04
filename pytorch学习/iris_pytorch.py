import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from sklearn.datasets import load_iris
import numpy as np


class IrisDataset(Dataset):
    def __init__(self):
        self.iris = load_iris()

    def __getitem__(self, index) -> T_co:
        return np.array(self.iris.data[index], dtype=np.float32), self.iris.target[index]

    def __len__(self):
        return 150