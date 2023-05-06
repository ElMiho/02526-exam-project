import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import skimage as ski
from paralleltomo import paralleltomo
from sklearn import linear_model
from generate_test_im import *

n = 50
num_pellets = 100
pellet_size = 1
att_coefs = [0.0157403151,0.0687329102,1]
im = generate_im(n,att_coefs,num_pellets,pellet_size)

theta = np.matrix(np.linspace(1 + 0, 1 + 180-2, 180 // 2))
p=30
A,theta,p,d = paralleltomo(N)
np.linalg.cond(A)

b = A @ x_input
lam = 0
model = Ridge(alpha = lam)

model.fit(A,b)
x_ridge = model.coef_
im_recov_ridge = np.reshape(x_ridge,(N,N))
plt.imshow(im_recov_ridge)
plt.show()