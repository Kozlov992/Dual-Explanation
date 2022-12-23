from dual_model import dual_MSE_explainer
from other_utils import linear_func_noisy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
np.random.seed(141516)
#%%Parameters
n = 1000
noise_std = 0.1
nbrs_num = 10
dual_samples_cnt = 30
coefficients = np.array([10, -20, -2, 3, 0, 0, 0])
#%%
X = np.random.uniform(0, 1, (n, 7))
f = linear_func_noisy(coefficients, noise_std)
nbrs = NearestNeighbors(n_neighbors=nbrs_num).fit(X)
dual_explainer = dual_MSE_explainer(f)

predictions_dual = []
for test_point in X:
    _, indices = nbrs.kneighbors(test_point.reshape(1, -1))
    points = np.append(X[indices[0]], test_point.reshape(1, -1), axis=0)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    dual_explainer.set_extreme_points(vertices)
    coefs = dual_explainer.explain(dual_samples_cnt)
    predictions_dual.append(coefs)

dual_coefficients = np.mean(np.array(predictions_dual), axis=0)
with np.printoptions(precision=2, suppress=True):
    print(dual_coefficients)