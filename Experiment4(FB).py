from sklearn.neighbors import KNeighborsRegressor
from other_models import gaussian_LIME_explainer
from dual_model import dual_MSE_explainer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import numpy as np
from other_utils import draw_explanations_comparison_chart
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
#%%Parameters
test_size = 100
nu = 0.5
perturbed_samples_count = 30
dual_samples_cnt = 30
cov = np.eye(4) * 0.1
nbrs_num = 10
nbrs_num_knn_reg = 10
#%% Data Preprocessing (Step 1)
data = pd.read_excel("CCPP_DataSet.ods", engine="odf")
standardScale = StandardScaler()
x = data.drop('PE', axis = 1)
y = data['PE']
#%% Data Preprocessing (Step 2)
x = standardScale.fit_transform(x)
nbrs = NearestNeighbors(n_neighbors=nbrs_num).fit(x)
chull = ConvexHull(x)
#%%Test data creation
test_dta = []
for simplice in chull.simplices[np.random.choice(chull.simplices.shape[0], test_size)]:
    simplice_vertices = simplice[np.random.choice(simplice.shape[0], 2)]
    lambda_, = np.random.uniform(0,1,1)
    new_point = lambda_ * x[simplice_vertices[0]] + (1 - lambda_) * x[simplice_vertices[1]]
    test_dta.append(new_point)
#%%
predict1 = KNeighborsRegressor(n_neighbors=nbrs_num_knn_reg).fit(x, y).predict
predict2 = RandomForestRegressor().fit(x,y).predict
models = [predict1, predict2]
models_MSE = []
last = []

for predict_func in models:
    dual_explainer = dual_MSE_explainer(predict_func)
    LIME_explainer = gaussian_LIME_explainer(predict_func)
    y_true = []
    y_predicted_dual = []
    y_predicted_lime = []
    for test_point in test_dta: 
        LIME_explainer.make_perturbed_background(test_point, cov, perturbed_samples_count)
        coefs_base = LIME_explainer.explain(nu)
        _, indices = nbrs.kneighbors(test_point.reshape(1, -1))
        points = np.append(x[indices[0]], test_point.reshape(1, -1), axis=0)
        hull = ConvexHull(points, qhull_options='QJ')
        vertices = points[hull.vertices]
        dual_explainer.set_extreme_points(vertices)
        coefs = dual_explainer.explain(dual_samples_cnt)
        last = coefs
        y_true.append( predict_func(test_point.reshape(1, -1))[0])
        y_predicted_dual.append(np.dot(coefs, test_point))
        y_predicted_lime.append(np.dot(coefs_base, test_point))
    dual_mse = metrics.mean_squared_error(y_true, y_predicted_dual)
    base_mse = metrics.mean_squared_error(y_true, y_predicted_lime)
    models_MSE.append((base_mse, dual_mse))
    
#%%
methods_labels = [r'LIME', 'dual method']
titles = ['kNN-regressor', 'RandomForest-regressor']
label = 'MSE'
draw_explanations_comparison_chart(methods_labels, models_MSE,
                                       label, titles, savefig_name='Exp4(FB).png',
                                       subplots_adjust_hspace=0.75)