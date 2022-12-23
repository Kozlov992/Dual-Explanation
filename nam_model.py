import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader 
from torch.nn import Module
import torch.nn.functional as F
import time
from typing import Tuple
import copy
device = torch.device('cpu')

class NAM(Module):      # Neural network 
    def _get_topology(self, use_nn_bias):
        return torch.nn.Sequential(
            torch.nn.Linear(1, 32, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16, use_nn_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1, use_nn_bias)
        )

    def __init__(self, dim: int, use_nn_bias: bool=True, lr: float = 0.001, l1=0, l2=0):
        super().__init__()
        self.dim = dim
        self.nn_list = []
        nn_params = []
        for _ in range(dim):
            nn = self._get_topology(use_nn_bias).to(device)
            self.nn_list.append(nn)
            nn_params += list(nn.parameters())
        self.optimizer = torch.optim.Adam(nn_params, lr, weight_decay=l2)
        self.loss_fn = F.mse_loss
        self.l1_coef = l1
        if l1 > 0:
            self.nn_params = nn_params

    def forward(self, x: torch.Tensor):
        #assert(x.dim() == 2 and x.shape[1] == self.dim)
        funcs_sum = 0
        for i in range(self.dim):
            funcs_sum += self.nn_list[i](x[:, i, None])
        return funcs_sum

    def predict(self, x: np.ndarray):
        self.eval()
        x_tens = torch.from_numpy(x).to(torch.float32).to(device)
        with torch.no_grad():
            res = self(x_tens)
        return res.cpu().numpy()
    
    def predict_shape_func(self, x: np.ndarray, func_num: int):
        self.eval()
        #assert(x.ndim == 2 and x.shape[1] == 1)
        nn = self.nn_list[func_num]
        x_tens = torch.from_numpy(x).to(torch.float32).to(device)
        with torch.no_grad():
            pred = nn(x_tens)
            return pred.cpu().numpy()
    
    def train_proc(self, x_train: np.ndarray, y_labels: np.ndarray, batch_size: int, epochs: int, val_data: Tuple[np.ndarray, np.ndarray] = None, patience: int = 10):
        start_time = time.time()
        if val_data is not None:
            #assert(val_data[0].ndim == 2)
            if val_data[1].ndim == 1:
             val_data[1] = val_data[1][:, np.newaxis]
            val_x, val_y = torch.from_numpy(val_data[0]).to(torch.float32).to(device), torch.from_numpy(val_data[1]).to(torch.float32).to(device)
            weights = copy.deepcopy(self.state_dict())
            cur_patience = 0
            def get_val_loss():
                with torch.no_grad():
                    val_pred = self(val_x)
                    val_loss = self.loss_fn(val_pred, val_y).item()
                return val_loss
            best_val_loss = get_val_loss()
        self.train()
        #assert(x_train.ndim == 2)
        if y_labels.ndim == 1:
            y_labels = y_labels[:, np.newaxis]
        x_tens = torch.from_numpy(x_train).to(torch.float32).to(device)
        y_tens = torch.from_numpy(y_labels).to(torch.float32).to(device)
        dataset = TensorDataset(x_tens, y_tens)
        data_loader = DataLoader(dataset, batch_size, shuffle=True)
        steps = len(data_loader)
        for e in range(1, epochs + 1):
            cur_loss = 0
            i = 0
            print(f'----- Epoch {e} -----')
            time_stamp = time.time()
            for x, y in data_loader:
                self.optimizer.zero_grad()
                pred = self(x)
                loss = self.loss_fn(pred, y)
                if self.l1_coef > 0:
                    penalty = 0
                    for param in self.nn_params:
                        penalty += torch.sum(torch.abs(param))
                    loss += self.l1_coef * penalty
                loss.backward()
                self.optimizer.step()
                cur_loss += loss.item()
                i += 1
                print(f'Loss: {round(cur_loss / i, 5)}, step {i}/{steps}', end='        \r')
            print()
            if val_data is not None:
                cur_patience += 1
                val_loss = get_val_loss()
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    weights = copy.deepcopy(self.state_dict())
                    cur_patience = 0
                print(f'val loss: {round(val_loss, 5)}, patience: {cur_patience}')
                if cur_patience >= patience:
                    print('Early stopping!')
                    self.load_state_dict(weights)
                    break
            print('time elapsed: ', round(time.time() - time_stamp, 4), ' sec.')
        print(f'Train is finished, {round(time.time() - start_time, 0)} sec. taken')
  