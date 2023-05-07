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
#import generate_test_functions

# Add noise
def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

# N = 50
# att_coefs = [1.07,380.2,1443.5]
# resized_image = generate_im(N,att_coefs,num_pellets=10,pellet_size=1)

test_image = np.load('testImage.npy')

down_sampling_factors = [200, 250, 300]
for d in down_sampling_factors:
    resized_image = ski.measure.block_reduce(test_image, block_size=d, func=np.mean)
    N, _N = resized_image.shape
    print(f"N: {N}")

    # What angels and how many rays pr angle
    theta = np.matrix(np.arange(1, 180, 5))
    p = 16

    A, _, _, _ = paralleltomo(N, theta, p, int(np.sqrt(2) * N))
    cond_A_1 = np.linalg.cond(A)
    print(f"cond(A): {cond_A_1}")

    # Image to vector
    x = resized_image.flatten(order="F")
    b = A @ x

    b_noise = add_noise_float(b, 0, 0.00001)

    # Alpha values for ridge and lasso
    alphas = [0, 1, 20]

    recoved_images = []
    plt.figure(f"N: {N}")
    # plt.figure(f"Ridge, down scaling factor: {d} (N = {N}), recovered")
    for idx, a in enumerate(alphas):
        plt.subplot(3, 3, idx+1)
        plt.title(f"alpha: {a}")

        model_ridge = sklin.Ridge(alpha=a, fit_intercept=False)
        model_ridge.fit(A, b_noise)
        im_recov_ridge = np.reshape(model_ridge.coef_, (N,N), order = "F")
        # recoved_images.append((im_recov_ridge, a))

        # kmeans_image, _ = validation.kmean_clust(im_recov_ridge)

        # Plotting the recovered image
        plt.axis("off")
        plt.imshow(im_recov_ridge)
    
    # plt.savefig(f".././images/ridge-N-{N}-recovered.png")


    # What angels and how many rays pr angle
    theta = np.matrix(np.arange(1, 180, 1))
    p = 48

    A, _, _, _ = paralleltomo(N, theta, p, int(np.sqrt(2) * N))
    cond_A_2 = np.linalg.cond(A)
    print(f"cond(A): {cond_A_2}")

    # Image to vector
    x = resized_image.flatten(order="F")
    b = A @ x

    b_noise = add_noise_float(b, 0, 0.00001)

    # Alpha values for ridge and lasso
    alphas = [0, 1, 20]

    for idx, a in enumerate(alphas):
        plt.subplot(3, 3, idx+4)
        plt.title(f"alpha: {a}")

        model_ridge = sklin.Ridge(alpha=a, fit_intercept=False)
        model_ridge.fit(A, b_noise)
        im_recov_ridge = np.reshape(model_ridge.coef_, (N,N), order = "F")
        recoved_images.append((im_recov_ridge, a))

        # kmeans_image, _ = validation.kmean_clust(im_recov_ridge)

        # Plotting the recovered image
        plt.axis("off")
        plt.imshow(im_recov_ridge)


    # plt.figure(f"Ridge, down scaling factor: {d} (N = {N}), kmean-cluster")
    for idx, (im_recov_ridge, a) in enumerate(recoved_images):
        plt.subplot(3, 3, idx+7)
        plt.title(f"alpha = {a}")
        # Plots the labels i.e. a computer can predict it e.g. 0,1,2,3 for 4 clusters
        kmeans_image, _ = validation.kmean_clust(im_recov_ridge)
        plt.axis("off")
        plt.imshow(kmeans_image)
        
    plt.savefig(f".././images/total-picture-{N}-condA1-{cond_A_1}-condA2-{cond_A_2}.png")

plt.show()