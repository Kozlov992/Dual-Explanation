from sklearn.linear_model import LinearRegression
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
import time
import copy


device = torch.device('cpu')

class gaussian_LIME_explainer():
    def __init__(self, explainable_function, dta=None, fit_intercept=False):
        self.regressor = LinearRegression(fit_intercept=fit_intercept)
        self.background_dta = dta
        self.explainbale_function = explainable_function
        self.explainble_point = None
        self.nu = None
        self.fit_intercept = fit_intercept
      
    def make_perturbed_background(self, mean, std, size):
        self.background_dta = np.random.multivariate_normal(mean, std, size)
      
    def set_background_dataset(self, dta):
        self.background_dta = dta
        
    def explain(self, point_to_explain, nu=1):
        self.nu = nu
        self.explainble_point = point_to_explain
        weights = self.calculate_weights()
        self.regressor.fit(self.background_dta, self.explainbale_function(self.background_dta), sample_weight=weights)
        if self.fit_intercept:
            return np.array([*self.regressor.coef_, self.regressor.intercept_])
        return self.regressor.coef_
        
    def calculate_weights(self):
        return np.exp(-np.linalg.norm(self.background_dta-self.explainble_point, axis=1) ** 2 / (2 * np.square(self.nu)))
    
    
#Wrapper class. A way to pass usual multivariate function as a "model" parameter into PyALE package functions.
class model_wrapper():
    def __init__(self, explainable_function):
        self.explainable_function = explainable_function
        
    def predict(self, X):
        return self.explainable_function(X.values)
    

class ConvNN(torch.nn.Module):
    def _get_conv_nn(img_shape):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 6 * 6, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.ReLU(),
        )
    
    def __init__(self):
        super().__init__()
        self.nn = self._get_conv_nn().to(device)

    def forward(self, x):
        return self.nn(x)
    
    def predict(self, x, batch=256):
        x_dl = DataLoader(TensorDataset(torch.from_numpy(x).to(torch.float32)), batch, shuffle=False)
        with torch.no_grad():
            output = []
            for cur_x in x_dl:
                output.append(self.nn(cur_x[0].to(device)).cpu().numpy())
        return np.concatenate(output, axis=0)

    def fit_proc(self, x, y, loss_fn, optimizer, batch_size, epochs, val_data, patience):
        start_time = time.time()
        if val_data is not None:
            val_x, val_y = torch.from_numpy(val_data[0]), torch.from_numpy(val_data[1])
            val_ds = TensorDataset(val_x, val_y)
            val_dl = DataLoader(val_ds, batch_size, shuffle=True)
            weights = copy.deepcopy(self.state_dict())
            cur_patience = 0
            def get_val_loss():
                val_loss = 0
                for val_x_batch, val_y_batch in val_dl:
                    val_x_batch, val_y_batch = val_x_batch.to(device), val_y_batch.to(device)
                    with torch.no_grad():
                        val_pred = self(val_x_batch)
                        val_loss += loss_fn(val_pred, val_y_batch).item()
                return val_loss / len(val_dl)
            best_val_loss = get_val_loss()
        self.train()
        x_tens = torch.from_numpy(x)
        y_tens = torch.from_numpy(y)
        dataset = TensorDataset(x_tens, y_tens)
        data_loader = DataLoader(dataset, batch_size, shuffle=True)
        steps = len(data_loader)
        for e in range(1, epochs + 1):
            cur_loss = 0
            i = 0
            print(f'----- Epoch {e} -----')
            time_stamp = time.time()
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = self(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
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
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.load_state_dict(torch.load(f, device))
