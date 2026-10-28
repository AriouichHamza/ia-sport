import yaml 
from utils.Arguments import arguments as args
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# pytorch
class Custom_Dataset(Dataset):
    def __init__(self, X, Y):
        """
        Initialize the custom dataset.

        Args:
            X (list or numpy.ndarray): Input data.
            Y (list or numpy.ndarray): Target data.
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert X to PyTorch tensor
        self.Y = torch.tensor(Y, dtype=torch.float32)  # Convert Y to PyTorch tensor

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            x (tensor): Input data for the sample.
            y (tensor): Target data for the sample.
        """
        x = self.X[idx]
        y = self.Y[idx]
        return x, y


# --- Chargement de la config API --
def load_api_key():
    path = args.config_path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["openai"]["api_key"]

# drop useless columns and rows
def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:(i + timesteps)])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)

 # for both multivariate & univariate split with y duplicates
def build_mutlivariate_training_data(x_dataset, past_history, forecast_horizon)->np.ndarray:
    # dataset: (num_samples,num_series)
    x = []
    y = []
    num_samples = len(x_dataset)
    num_possible_split_iter = ((num_samples - past_history) + 1) - forecast_horizon
    # start_index_forecast  = past_history   #first start index of y labels
    # end_index_forecast = (start_index_forecast + forecast_horizon) - 1
    for i in range(0, num_possible_split_iter):
        x.append(x_dataset[i:i + past_history])
        y.append(x_dataset[i + past_history: (i + past_history) + forecast_horizon])
    return  np.asarray(x), np.asarray(y)


def convert_to_numpy_arry(datasets):
    tensors = {name: np.asarray(ds) for name, ds in datasets.items()}
    return tensors

def convert_to_pytorch_tensor(datasets):
    tensors = {name: torch.tensor(ds, dtype=torch.float32) for name, ds in datasets.items()}
    return tensors


def convert_from_3D_to_2D(array):
    arr_shape = array.shape
    array_ = array.reshape(arr_shape[0]*arr_shape[1], arr_shape[2])
    return array_

#add extra (third) dim to 2D datasets
def add_extra_dim(datasets, ndim):
    datasets_ = {}
    for name, ds in datasets.items():
        if ds.ndim != ndim:
            ds = ds[..., None] #add extra dim at the end
        datasets_[name] = ds
    return datasets_

def create_custom_datasets(datasets):
    # cumstom datasets
    train_dataset = Custom_Dataset(datasets["X_train"], datasets["y_train"])
    #validation_dataset = self.Custom_Dataset(datasets["X_val"], datasets["y_val"])
    test_dataset = Custom_Dataset(datasets["X_test"], datasets["y_test"])
    datasets_ = {
        "train_dataset": train_dataset,
        # "valid_dataset": validation_dataset,
        "test_dataset": test_dataset
    }
    return datasets_

def get_dataloaders(datasets, batch_size, shuffle):
    
    dataloaders = {
    'train_dataloader': DataLoader(datasets['train_dataset'], batch_size, shuffle=shuffle, drop_last=False),
    # 'valid_dataloader' : DataLoader(datasets['valid_dataset'], batch_size, drop_last=False),
    'test_dataloader' : DataLoader(datasets['test_dataset'], batch_size, drop_last=False)
    }
    return dataloaders 