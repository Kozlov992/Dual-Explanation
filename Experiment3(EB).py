import numpy as np
import pandas as pd
from dual_model import dual_func
from dual_model_utils import run_dual_model
np.random.seed(169)
#%%
n = 1000
X = np.array([[-1.,-1.], [0.,2.], [1.,0.]]) #polytope vertices
func_X = lambda x: np.sign(x[:,1]) + 0.7 * np.sign(x[:,0]) 
d_func = dual_func(X, func_X)
lambdas = np.random.dirichlet(np.ones(3), 1000)
z = d_func(lambdas)
X_dta = lambdas @ X
params_NAM = {'bias': True, 'learning_rate': 0.0005, 'a_1':0, 'a_2':0,
              'epochs': 300, 'batch_size': 128}
params_reg = {'regression_type': 'MSE'}
params_ALE = {'func': d_func, 'grid_size': 100}
model_type = ['ALE', 'lin_reg', 'NAM']
model_params = [params_ALE, params_reg, params_NAM]
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