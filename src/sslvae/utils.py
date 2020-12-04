import torch
from torch.utils.data import  Subset, Dataset

class HConcatDataset(Dataset):
    def __init__(self, labelled, unlabelled):
        self.labelled = labelled
        self.unlabelled = unlabelled

    def __getitem__(self, item):
        x_l, y_l = self.labelled[item]
        x_u, y_u = self.unlabelled[item]

        one_hot_l = [0] * 10
        one_hot_l[y_l] = 1
        y_l = torch.tensor(one_hot_l, dtype=torch.float)

        one_hot_u = [0] * 10
        one_hot_u[y_u] = 1
        y_u = torch.tensor(one_hot_u, dtype=torch.float)

        return (x_l, y_l, x_u, y_u)

    def __len__(self):
        return min(len(d) for d in [self.labelled, self.unlabelled])


def make_mnist_semi_super(data, pct=1.0, train=False, val=False):
    if pct > 1:
        raise Exception("Percentage should be a value between 0-1")

    if train and val:
        raise Exception('Cannot split train and val at the same time')

    # split train data
    n = int(len(data) * (pct))
    if train:
        data.data = data.data[:n]
        data.targets = data.targets[:n]
    elif val:
        data.data = data.data[-n:]
        data.targets = data.targets[-n:]
    else:
        pass

    # Make dataset semi-supervised
    ids = torch.randperm(len(data)).tolist()
    nl = len(data) // 2
    nu = len(data) - nl

    ids_l = ids[:nl]
    ids_u = ids[nl:nl + nu]
    data_l = Subset(data, ids_l)
    data_u = Subset(data, ids_u)

    dataset = HConcatDataset(data_l, data_u)
    return dataset

