from typing import Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

class CancerDataset(Dataset):
    def __init__(self, X=None, y=None, train=True):
        if X is None or y is None:
            cancer_data = load_breast_cancer()
            X = cancer_data.data
            y = cancer_data.target
        if train:
            self.X, _, self.y, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            _, self.X, _, self.y = train_test_split(X, y, test_size=0.2, random_state=42)
        self.dataset = CancerDataset_supported(self.X, self.y)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
class CancerDataset_supported(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class BostonHousingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.X = np.array(self.data.iloc[:, :-1], dtype=np.float32)
        self.y = np.array(self.data.iloc[:, -1], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
    
class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y
    
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = datasets.MNIST(root=root, train=train, transform=transform, download=True)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x.view(-1), y

    def __len__(self):
        return len(self.data)



class RotatedCIFAR100(datasets.CIFAR100):
    """ CIFAR100 rotated by fixed amount using random seed """
    # taken from ntk-marglik implimentation 
    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class WikiText2Dataset(Dataset):
    def __init__(self, split='train', batch_size=20, bptt=35):
        self.split = split
        self.batch_size = batch_size
        self.bptt = bptt

        self.train_iter, self.val_iter, self.test_iter = WikiText2()
        self.vocab = self.build_vocab()
        self.train_data = self.data_process(self.train_iter)
        self.val_data = self.data_process(self.val_iter)
        self.test_data = self.data_process(self.test_iter)

        

        self.train_data = self.batchify(self.train_data, self.batch_size)
        self.val_data = self.batchify(self.val_data, self.batch_size)
        self.test_data = self.batchify(self.test_data, self.batch_size)

        if self.split == 'train':
            self.data = self.train_data
        elif self.split == 'val':
            self.data = self.val_data
        elif self.split == 'test':
            self.data = self.test_data

    def __len__(self):
        return (self.data.size(0) // self.bptt) - 1

    def __getitem__(self, idx):
        start_index = idx * self.bptt
        end_index = start_index + self.bptt
        if end_index >= self.data.size(0):
            data = self.data[start_index:]
            target = self.data[start_index+1:]
        else:
            data = self.data[start_index:end_index]
            target = self.data[start_index+1:end_index+1]

        data = data.reshape(-1)
        target = target.reshape(-1)

        return data, target


    def build_vocab(self):
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.train_iter),
            specials=['<unk>']
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        return self.vocab

    def data_process(self, raw_text_iter):
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        seq_len = self.bptt
        num_batches = (data.size(0) - 1) // (bsz * seq_len)
        data = data[:num_batches * bsz * seq_len]
        data = data.view(bsz, -1).t().contiguous()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return data.to(device)

    def get_batch(self, source, i):
        seq_len = self.bptt
        data = source[i:i + seq_len].t().contiguous().reshape(-1)
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

