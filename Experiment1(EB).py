import numpy as np
import pandas as pd
from dual_model_utils import run_dual_model, plot_shape_functions, plot_shape_functions_single_plot
#%%Parameters
func = lambda x: (15 * x[:,0] + 22 * x[:,1] + 40 *(1 - x[:,3]) * np.sin(np.math.pi * x[:,3]))
dim = 6
n = 2000
params_NAM = {'bias': True, 'learning_rate': 0.0005, 'a_1':1e-4, 'a_2':1e-4,
              'epochs':300, 'batch_size': 128}
params_reg = {'regression_type': 'MSE'}
params_ALE = {'func': func, 'grid_size': 100}
#%%
lambdas = np.random.dirichlet(np.ones(dim), n)
z = func(lambdas)
model_type = ['ALE', 'lin_reg', 'NAM']
model_params = [params_ALE, params_reg, params_NAM]
#model_params = [params_NAM]
model_importances = {}
models = {}
for i in range(len(model_type)):
    m_type = model_type[i]
    m_params = model_params[i]
    importances, model = run_dual_model(lambdas, z, m_type, m_params)
    models[m_type] = model
    model_importances[m_type] = importances
    
for key, value in model_importances.items():
    print(key,value)
print(pd.DataFrame(model_importances).T)
#%%Visualization
plot_shape_functions(models['NAM'], lambdas)
plot_shape_functions_single_plot(models['NAM'], lambdas, plot_dimensions=(2, 3),
                                 hspace=0.6, wspace=0.2, figsize=(10,4), horizontal_window=3)