import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from other_models import ConvNN
from other_utils import split_dataset, get_images_ds, save_images, MAE, load_images
from dual_model_utils import run_dual_model, plot_shape_functions, plot_shape_functions_single_plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device('cpu')
np.random.seed(42)
def train_image_regressor():
    batch_size = 128
    epochs = 200
    patience = 10
    tr_img, tr_lbl, val_img, val_lbl, test_img, test_lbl = split_dataset(*get_images_ds('./UTKFace/'))
    save_images('train_data', tr_img, tr_lbl)
    save_images('test_data', test_img, test_lbl)
    nn = ConvNN()
    opt = torch.optim.Adam(nn.parameters())
    loss = MAE
    nn.fit_proc(tr_img, tr_lbl, loss, opt, batch_size, epochs, (val_img, val_lbl), patience)
    print('test MAE:', np.mean(np.abs(test_lbl - nn.predict(test_img))))
    nn.save('nn_weights')

#%%Model loading; data preprocessing
nn = ConvNN().to(device)
#train_image_regressor()    #in case retraining is preffered
nn.load('nn_weights')
func = lambda x: nn.predict(x.reshape(-1, 128, 128)[:,np.newaxis,:,:]).reshape(-1,)
test_img, test_lbl = load_images('test_data_img', 'test_data_lbl')
train_img, train_lbl = load_images('train_data_img', 'train_data_lbl')
X_train = train_img[:,0,:,:].reshape(-1, 128 * 128)
X_test = test_img[:,0,:,:].reshape(-1, 128 * 128)
#destination_folder = ''
#%%Parametrers
N = 10  #neighbours
n = 3000
ind = 1013  #indice of explained instance (from test data)
params_NAM = {'bias': True, 'learning_rate': 0.0005, 'a_1':0, 'a_2':0,
              'epochs': 100, 'batch_size': 128}
#model_type = ['ALE', 'lin_reg', 'NAM']
model_type = ['NAM']
model_params = [params_NAM]
#%%
#metrics: 'l1', 'l2', 'cosine', 'WEuclDis1', 'WEuclDis2', 'WEuclDis3'
WEuclDis1 = lambda x, y: 0.5 * np.linalg.norm(x[:-1] - y[:-1]) + 0.5 * np.abs(x[-1] - y[-1])
WEuclDis2 = lambda x, y: 0.7 * np.linalg.norm(x[:-1] - y[:-1]) + 0.3 * np.abs(x[-1] - y[-1])
WEuclDis3 = lambda x, y: np.sqrt(0.25 * np.linalg.norm(x[:-1] - y[:-1]) ** 2 + 0.75 * (x[-1] - y[-1]) ** 2)

metric = 'cosine'

if metric in ('l1', 'l2', 'cosine'):
    nbrs = NearestNeighbors(n_neighbors=N, metric=metric).fit(X_train)
    _, indices = nbrs.kneighbors(X_test[ind].reshape(1,-1))
else:
    joined_X = np.hstack((X_train, train_lbl))
    nbrs = NearestNeighbors(n_neighbors=N, metric=metric).fit(joined_X)
    _, indices = nbrs.kneighbors(np.hstack((X_test[ind].reshape(1,-1), func(X_test[ind].reshape(1,-1))[:,np.newaxis])))
    
lambdas_raw = np.random.dirichlet(np.ones(N + 1), 3000)
X_vertices = np.append(X_train[indices[0]], X_test[ind].reshape(1,-1),axis=0)
X_nam_train = lambdas_raw @ X_vertices
z = func(X_nam_train)
lambdas = lambdas_raw[:,:-1]
#%%Explained instance
plt.imshow(X_test[ind].reshape(128,128), cmap='gray')
predicted_age = func(X_test[ind].reshape(1,-1))[0]
real_age = int(test_lbl[ind][0])
plt.xlabel(f'Age: {real_age}\n Predicted age: {predicted_age:.1f}')
plt.title('Explained point')
plt.grid(False)
plt.xticks([])
plt.yticks([])
#plt.savefig(destination_folder + 'Explained_point'+ '.png', dpi=220, bbox_inches='tight')
plt.show()
#%%
importances, model = run_dual_model(lambdas, z, model_type[0], model_params[0])
#%%
print(pd.DataFrame(importances).T)
#%%
importances = importances.reshape(1,-1)
plt.imshow((importances @ X_train[indices[0]])[0].reshape(128,128), cmap='gray')
predicted_age = func(importances @ X_train[indices[0]])[0]
weighted_age = np.dot(importances[0], train_lbl[indices[0]].ravel())
plt.xlabel(f'Predicted age: {predicted_age:.1f}\n Weighted age: {weighted_age:.1f}')
plt.title('Explainer point')
plt.grid(False)
plt.xticks([])
plt.yticks([])
#plt.savefig(destination_folder + 'Explainer_point'+ '.png', dpi=220, bbox_inches='tight')
plt.show()
#%%
plot_shape_functions(model, lambdas, horizontal_window=1)
plot_shape_functions_single_plot(model, lambdas, plot_dimensions=(2, 5),
                             hspace=0.55, wspace=0.35, figsize=(12,6), horizontal_window=1)
#%%
k = 0
fig, ax = plt.subplots(2, 5, figsize=(10,5))
predicted_ages = func(X_train[indices[0]])
real_ages = train_lbl[indices[0]].ravel()
for i in range(2):
    for j in range(5):
        predicted_age = predicted_ages[k]
        real_age = real_ages[k]
        ax[i,j].imshow(X_train[indices[0]][k].reshape(128,128), cmap='gray')
        ax[i,j].set_xlabel(f'Age: {real_age:.0f}\nPredicted age: {predicted_age:.1f}')
        #ax[i,j].set_ylabel('$y$')
        #ax[i, j].set_ylim([min_z -1, max_z + 1])
        ax[i,j].set_title(f'Example {k+1}')
        ax[i,j].grid(False)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        k += 1
plt.subplots_adjust(hspace=0.55, wspace=0.35)
#plt.savefig(destination_folder + 'examples_all'+ '.png', dpi=220, bbox_inches='tight')
plt.show()