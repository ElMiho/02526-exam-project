import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import skimage as ski
from paralleltomo import paralleltomo
import sklearn.linear_model as sklin
from generate_test_functions import *

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
#Plot of pixels as a function of alpha
n_alphas = 10
alphas = np.logspace(10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = sklin.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(A, b)
    ridge.coef_ = ridge.coef_ / np.max(ridge.coef_)
    coefs.append(ridge.coef_)


b = A @ x_input
lam = 0
model = sklin.Ridge(alpha = lam)

model.fit(A,b)
x_ridge = model.coef_
im_recov_ridge = np.reshape(x_ridge,(N,N))
plt.imshow(im_recov_ridge)
plt.show()