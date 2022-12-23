import numpy as np
import pandas as pd
from other_models import model_wrapper
from nam_model import NAM
import torch
from sklearn.linear_model import LinearRegression
from PyALE import ale
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cpu')

class dual_func():
    def __init__(self, vertices, explainable_func):
        self.vertices = vertices
        self.explainable_func = explainable_func
        
    def __call__(self, lambdas):
        _generated_set = lambdas @ self.vertices
        return self.explainable_func(_generated_set)
    

def run_dual_model(lambdas, z, model_type, model_params, importances_precision=3):
    importances, model = None, None
    if model_type == 'NAM':
        use_bias = model_params['bias']
        learning_rate = model_params['learning_rate']
        epochs = model_params['epochs']
        batch_size = model_params['batch_size']
        a_1 = model_params['a_1']
        a_2 = model_params['a_2']
        dta = lambdas
        model = NAM(lambdas.shape[-1], use_bias, learning_rate, a_1, a_2).to(device)
        model.train_proc(dta, z[:, np.newaxis], batch_size, epochs)
        
        #calculate importances
        importances = []
        for i in range(dta.shape[-1]):
            single_coord_prediction = model.predict_shape_func(dta[:,i][:, np.newaxis], i).ravel()
            importances.append(np.std(single_coord_prediction))
            
        importances = np.array(importances)
        importances = np.around(importances / np.linalg.norm(importances, 1), importances_precision)
    elif model_type == 'lin_reg':
        model = LinearRegression(fit_intercept=False)
        dta = lambdas
        model.fit(dta, z)
        importances = np.around(np.abs(model.coef_) / np.linalg.norm(model.coef_, ord=1), importances_precision)
    elif model_type == 'ALE':
        model = []
        importances = []
        func = model_params['func']
        grid_size = model_params['grid_size']
        dta = pd.DataFrame(lambdas, columns=[f'$\lambda_{i+1}$' for i in range(lambdas.shape[-1])])
        for i in range(lambdas.shape[-1]):
            ale_eff = ale(X=dta, model=model_wrapper(func), feature=[f'$\lambda_{i+1}$'], grid_size=grid_size, include_CI=False)
            #plt.show()
            model.append(ale_eff)
            importances.append(np.std(ale_eff['eff'].values))
        importances = np.array(importances)
        importances = np.around(importances / np.linalg.norm(importances, 1), importances_precision)
    return importances, model

def shape_functions_EDA(nam_model, lambdas):
    z_plot = []
    lambda_plot = []
    min_z, max_z = None, None
    for i in range(lambdas.shape[-1]):
        lower, upper = min(lambdas[:,i]), max(lambdas[:,i])
        lambda_plot.append(np.linspace(lower, upper, 1000))
        z_plot.append(nam_model.predict_shape_func(lambda_plot[-1][:, np.newaxis], i).ravel())
        if i == 0:
            min_z, max_z = min(z_plot[-1]), max(z_plot[-1])
        else:
            min_z, max_z = min(min(z_plot[-1]), min_z), max(max(z_plot[-1]), max_z)
    return min_z, max_z, lambda_plot, z_plot
    
def plot_shape_functions(nam_model, lambdas, dpi=220, horizontal_window = 1):
    min_z, max_z, lambda_plot, z_plot = shape_functions_EDA(nam_model, lambdas)
    for i in range(lambdas.shape[-1]):
        plt.plot(lambda_plot[i], z_plot[i], linewidth=1.5, color='green')
        plt.xlabel(f'$\lambda_{i+1}$')
        plt.ylabel('$y$')
        plt.ylim([min_z - horizontal_window, max_z + horizontal_window])
        plt.title(f'$h_{i+1}(\lambda_{i+1})$')
        #plt.savefig(f'h_{i+1}'+ '.png', dpi=dpi)
        plt.show()
    
def plot_shape_functions_single_plot(nam_model, lambdas, plot_dimensions,
                                     dpi=220, horizontal_window = 0.05, hspace=0.5,
                                     wspace=0., figsize=(10, 3)):
    min_z, max_z, lambda_plot, z_plot = shape_functions_EDA(nam_model, lambdas)
    fig, ax = plt.subplots(*plot_dimensions, figsize=figsize)
    k = 0
    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            lower, upper = min(lambdas[:,k]), max(lambdas[:,k])
            lambda_plot = np.linspace(lower, upper, 1000)
            z_plot = nam_model.predict_shape_func(lambda_plot[:, np.newaxis], k).ravel()
            ax[i, j].plot(lambda_plot, z_plot, linewidth=1.5, color='green')
            ax[i, j].set_xlabel(f'$\lambda_{k+1}$')
            #ax[i,j].set_ylabel('$y$')
            ax[i, j].set_title(f'$h_{k+1}(\lambda_{k+1})$')
            ax[i, j].set_ylim([min_z - horizontal_window, max_z + horizontal_window])
            k += 1
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    #plt.savefig('h_all'+ '.png', dpi=dpi, bbox_inches = 'tight')
    plt.show()