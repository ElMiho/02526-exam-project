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
            y = np.abs([x-attenuations[0],x-attenuations[1],x-attenuations[2]])
            idx = np.argmin(y)
            recovered_image[i,j] = attenuations[idx]

n = 50
num_pellets = 100
pellet_size = 1
att_coefs = [0.0157403151,0.0687329102,1]
im = generate_im(n,att_coefs,num_pellets,pellet_size)

x_input = im.flatten(order = "F")

theta = np.matrix(np.linspace(1 + 0, 1 + 180-2, 180 // 2))
p=30
A, _,_ ,_ = paralleltomo(n,theta,p)
print(f"Condition number {np.linalg.cond(A)}")

b = A @ x_input

#Plot of pixels as a function of alpha
n_alphas = 10
alphas = np.logspace(10, -2, n_alphas)
coefs = []
for a in alphas:
    model = sklin.Ridge(alpha=a, fit_intercept=False)
    model.fit(A, b)
    model.coef_ = model.coef_ / np.max(model.coef_)
    coefs.append(model.coef_)