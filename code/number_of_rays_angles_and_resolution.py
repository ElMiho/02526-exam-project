from paralleltomo import paralleltomo
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import skimage as ski
from scipy import linalg
from tqdm import tqdm
import time
# import cvxpy as cp
import sklearn.linear_model as sklin
import validation

test_image = np.load('testImage.npy')

down_sampling_factor = 150
resized_image = ski.measure.block_reduce(test_image, block_size=down_sampling_factor, func=np.mean)
N, _N = resized_image.shape
print(f"N: {N}")

# What angels and how many rays pr angle
theta = np.matrix(np.arange(1, 180))
p = 48

A, _, _, _ = paralleltomo(N, theta, p, int(np.sqrt(2) * N))
cond_A = np.linalg.cond(A)
print(f"cond(A): {cond_A}")

# Image to vector
x = resized_image.flatten(order="F")
b = A @ x

# Add noise
def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

b_noise = add_noise_float(b, 0, 0.0001)

# Alpha values for ridge and lasso
alphas = [0, 1/2, 1, 5, 
          10, 20, 25,
          40, 50, 60]

plt.figure("Different alpha values - ridge")
for idx, a in enumerate(alphas):
    plt.subplot(4, 4, idx+1)
    plt.title(f"alpha = {a}")

    model_ridge = sklin.Ridge(alpha=a, fit_intercept=False)
    model_ridge.fit(A, b_noise)
    im_recov_ridge = np.reshape(model_ridge.coef_,(N,N), order = "F")

    kmeans_image, _ = validation.kmean_clust(im_recov_ridge)

    plt.imshow(kmeans_image)

# plt.show()

plt.figure("Different alpha values - lasso")
for idx, a in enumerate(alphas):
    plt.subplot(4, 4, idx+1)
    plt.title(f"alpha = {a}")

    model_lasso = sklin.Lasso(alpha=a, fit_intercept=False)
    model_lasso.fit(A, b_noise)
    im_recov_lasso = np.reshape(model_lasso.coef_,(N,N), order = "F")

    kmeans_image, _ = validation.kmean_clust(im_recov_lasso)

    plt.imshow(kmeans_image)

plt.show()