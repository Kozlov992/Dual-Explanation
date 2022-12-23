import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from dual_model import dual_MSE_explainer
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(141516)
#%%Parameters
n = 100
nbrs_num = 6
limits1 = [0, 1]
limits2 = [15, 16]
dual_samples_count = 30
explainable_func = lambda X: - np.square(X[:, 0]) + 2 * X[:, 1] + np.random.normal(loc=0, scale=0.05, size=X.shape[0])
#%%
dual_explainer = dual_MSE_explainer(explainable_func, dual_samples_count)

samples1 = np.random.uniform(*limits1, (n,2))
samples2 = np.random.uniform(*limits2, (n,2))
samples = [samples1, samples2]

nbrs1 = NearestNeighbors(n_neighbors=nbrs_num).fit(samples1)
nbrs2 = NearestNeighbors(n_neighbors=nbrs_num).fit(samples2)
nbrs = [nbrs1, nbrs2]

average_feature_importances1 = np.zeros(2)
average_feature_importances2 = np.zeros(2)
average_feature_importances = [average_feature_importances1, average_feature_importances2]

cases_number = len(samples)
for i in range(cases_number):    
    for test_point in samples[i]:
        _, indices = nbrs1.kneighbors(test_point.reshape(1, -1))
        points = samples[i][indices[0]]
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        dual_explainer.set_extreme_points(vertices)
        average_feature_importances[i] += dual_explainer.explain(dual_samples_count=20) / n
    
#%%Visualization
importances = [average_feature_importances[i] / np.linalg.norm(average_feature_importances[i]) for i in range(cases_number)]
fig, ax = plt.subplots(1, cases_number)
for i in range(cases_number):
    _labels = [r'$x_1$', r'$x_2$']
    vert_pos = np.arange(len(_labels))
    ax[i].bar(vert_pos, importances[i], align='center', color='tab:orange')
    ax[i].set_xticks(vert_pos, labels=_labels)
    ax[i].set_ylim([-1,1])
ax[0].set_ylabel('Normalized feature importance')
plt.subplots_adjust(wspace=0.3)
plt.show()