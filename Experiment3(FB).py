from sklearn.neighbors import KNeighborsRegressor
from other_models import gaussian_LIME_explainer
from dual_model import dual_MSE_explainer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import numpy as np
from other_utils import circle_uniform, draw_explanations_comparison_chart
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
np.random.seed(141516)
#%%Parameters
R = 2
r = 1.9
n = 100
N = 400
nu = 0.01
perturbed_samples_count = 30
dual_samples_cnt = 30
cov = np.eye(2) * 0.05
nbrs_num = 6
nbrs_num_knn_reg = 6
#%%Dataset visualization
explainable_func = lambda x1, x2: x1 * x1 + x2 * x2 + np.random.normal(0, 0.05, x1.shape[0])
x_train, y_train = circle_uniform(0, R, N)
x_test, y_test = circle_uniform(r, R, n)
for render in ('depth', 'no_depth'):
    for dataset_type in ('Train set', 'Test set'):
        circle = plt.Circle((0, 0), R, fill=False, color='tab:blue', linestyle='--')
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_aspect(1)
        ax.add_artist(circle)
        if dataset_type == 'Train set':
            x, y = x_train, y_train
        else:
            x, y = x_test, y_test
            inner_circle = plt.Circle((0, 0), r, fill=False, color='tab:blue', linestyle='--')
            ax.add_artist(inner_circle)
        if render == 'depth':
            plt.scatter(x, y, c=explainable_func(x,y), s=10, cmap='winter', vmin=0)
            plt.colorbar()
        else:
            plt.scatter(x, y, s=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(dataset_type)
        #plt.savefig()
        plt.show()
#%%
explainable_func = lambda X: 2 * X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1] + np.random.normal(0, 0.05, X.shape[0])
dta = np.array([x_train, y_train]).T
dta_test = np.array([x_test, y_test]).T
values = explainable_func(dta)
rf = RandomForestRegressor().fit(dta, values)
knn_reg = KNeighborsRegressor(n_neighbors=nbrs_num_knn_reg).fit(dta, values)
models = [knn_reg, rf]
models_MSE = []
values_test = explainable_func(dta_test)
nbrs = NearestNeighbors(n_neighbors=nbrs_num).fit(dta)

for model in models:
    f = lambda X: model.predict(X).flatten()
    circle_dual_explainer = dual_MSE_explainer(f)
    
    predictions_dual = []
    for test_point in dta_test:
        _, indices = nbrs.kneighbors(test_point.reshape(1, -1))
        points = np.append(dta[indices[0]], test_point.reshape(1, -1), axis=0)
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        circle_dual_explainer.set_extreme_points(vertices)
        coefs = circle_dual_explainer.explain(dual_samples_cnt)
        predictions_dual.append(np.dot(coefs, test_point))
        
    dual_mse = mean_squared_error(predictions_dual, values_test)
    
    
    circle_test_LIME_explainer = gaussian_LIME_explainer(f)
    
    base_predictions = []
    for test_point in dta_test:
        circle_test_LIME_explainer.make_perturbed_background(test_point, cov, perturbed_samples_count)
        coefs_base = circle_test_LIME_explainer.explain(test_point, nu=nu)
        base_predictions.append(np.dot(coefs_base, test_point))
    base_mse = mean_squared_error(base_predictions, values_test)
    models_MSE.append((base_mse, dual_mse))
#%%
methods_labels = [r'LIME', 'dual method']
titles = ['kNN-regressor', 'Random-Forest Regressor']
label = 'MSE'
draw_explanations_comparison_chart(methods_labels, models_MSE,
                                       label, titles, savefig_name='Exp3(FB).png',
                                       subplots_adjust_hspace=0.75)