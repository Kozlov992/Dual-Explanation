import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from PIL import Image

class linear_func():
    def __init__(self, coefficients):
        self.coefficients = coefficients
    
    def __call__(self, X):
        return X @ self.coefficients


class linear_func_noisy():
    def __init__(self, coefficients, std):
        self.std = std
        self.coefficients = coefficients
    
    def __call__(self, X):
        return X @ self.coefficients + np.random.normal(loc=0, scale=self.std, size=X.shape[0])
    

def circle_uniform(r=1,R=2,N=100):
    rad = np.sqrt(np.random.uniform(r * r, R * R, N))
    theta = np.random.uniform(0, 2 * np.math.pi, N)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return x,y


def as_cvx_cmb(x, points):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success, lp.x
 

def find_point_cloud_extreme_points_linprog(point_cloud):
    result = []
    cloud_size = len(point_cloud)
    for i in range(cloud_size):
        x = point_cloud[i]
        points = point_cloud[np.arange(cloud_size)!=i]
        if not as_cvx_cmb(x, points)[0]:
            result.append(x)
    return np.array(result)

def draw_explanations_comparison_chart(methods_labels, explanations_fidelities,
                                       label, titles, savefig_name=None,
                                       subplots_adjust_hspace=0.5, dpi=320):
    N = len(methods_labels)
    vert_pos = np.arange(N)
    fig, ax = plt.subplots(N, 1)
    for i in range(N):
        ax[i].barh(vert_pos, explanations_fidelities[i], align='center', color='tab:blue')
        ax[i].set_yticks(vert_pos, labels=methods_labels)
        ax[i].set_xlabel(label)
        ax[i].set_title(titles[i])
    plt.subplots_adjust(hspace=subplots_adjust_hspace)
    if savefig_name != None:
        fig.savefig(savefig_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def MAE(y_pred, y_labels):
    return torch.mean(torch.abs(y_pred - y_labels))

def get_images_ds(images_path):
    images_path = Path(images_path)
    labels = []
    images = []
    for img_path in images_path.iterdir():
        labels.append(int(img_path.name.split('_')[0]))
        pil_img = Image.open(str(img_path)).convert('L')
        pil_img.thumbnail((128, 128))
        images.append(np.asarray(pil_img, dtype=np.float32)[np.newaxis, ...])
    images = np.stack(images, 0)
    images = images / 255
    labels = np.asarray(labels, dtype=np.float32)[:, np.newaxis]
    return images, labels

def save_images(filename, images, labels):
    with open(filename + '_img', 'wb') as f:
        np.save(f, images)
    with open(filename + '_lbl', 'wb') as f:
        np.save(f, labels)

def load_images(img_filename, lbl_filename):
    with open(img_filename, 'rb') as f:
        images = np.load(img_filename)
    with open(lbl_filename, 'rb') as f:
        labels = np.load(lbl_filename)
    return images, labels

def split_dataset(images, labels, train_part = 0.75, val_part = 0.15, test_part = 0.1):
    idx = np.arange(images.shape[0])
    rng = np.random.default_rng()
    rng.shuffle(idx)
    tr_num = int(train_part * images.shape[0])
    val_num = int(val_part * images.shape[0])
    tr_img, tr_lbl = images[idx[:tr_num]], labels[idx[:tr_num]]
    val_img, val_lbl = images[idx[tr_num:tr_num+val_num]], labels[idx[tr_num:tr_num+val_num]]
    test_img, test_lbl = images[idx[tr_num+val_num:]], labels[idx[tr_num+val_num:]]
    return tr_img, tr_lbl, val_img, val_lbl, test_img, test_lbl
