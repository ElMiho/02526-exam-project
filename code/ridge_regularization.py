import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import skimage as ski
from paralleltomo import paralleltomo
import sklearn.linear_model as sklin
from generate_test_functions import *

def likelihood(recovered_image, attenuations):
    n,m = recovered_image.shape
    for i in range(n):
        for j in range(m):
            x = recovered_image[i,j]
            y = np.abs([x-attenuations[0],x-attenuations[1],x-attenuations[2],x-attenuations[3]])
            idx = np.argmin(y)
            recovered_image[i,j] = attenuations[idx]
    return recovered_image

def poisson_noise(b, lamb=1):
    return b + np.random.poisson(lamb, size=b.shape)

def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

n = 50
num_pellets = 100
pellet_size = 1

# Air, Wood, Bismuth, Iron at 16.4 keV
att_coefs = [0,1.0646,147.8,48.3665]
im = generate_im(n,att_coefs,num_pellets,pellet_size)

x_input = im.flatten(order = "F")

theta = np.matrix(np.linspace(1 + 0, 1 + 180-2, 180 // 2))
p=30
A, _,_ ,_ = paralleltomo(n,theta,p)
print(f"Condition number {np.linalg.cond(A)}")

b = A @ x_input

b_poisson = poisson_noise(b,lamb = 1)


# Plot of pixels as a function of alpha ||Ax-b||_2 + alpha*||x||_2
n_alphas = 10
alphas = np.logspace(10, -2, n_alphas)
no_noise_predict = []
noise_predict = []
for a in alphas:
    model = sklin.Ridge(alpha=a, fit_intercept=False)

    model.fit(A, b)
    im_recov_no_noise = np.reshape(model.coef_,(n,n))

    model.fit(A,b_poisson)
    im_recov_noise = np.reshape(model.coef_,(n,n))

    no_noise_predict.append(likelihood(im_recov_no_noise,att_coefs))
    noise_predict.append(likelihood(im_recov_noise,att_coefs))

plt.figure(1)
# Plot images
for i in range(n_alphas):
    plt.subplot(4,3,i+1)
    plt.imshow(np.asarray(no_noise_predict[i]))

plt.savefig(";))")
plt.close()

plt.figure(2)
for i in range(n_alphas):
    plt.subplot(4,3,i+1)
    plt.imshow(np.asarray(noise_predict[i]))

plt.savefig(";)))")
plt.close()