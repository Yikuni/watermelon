from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from sklearn.datasets import load_iris


class IrisDataset(Dataset):
    def __init__(self):
        self.iris = load_iris()

    def __getitem__(self, index) -> T_co:
        return self.iris.data[index], self.iris.target[index]

    def __len__(self):
        return 150