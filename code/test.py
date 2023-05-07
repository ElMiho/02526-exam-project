import matplotlib.pyplot as plt
import numpy as np
from paralleltomo import paralleltomo
import sklearn.linear_model as sklin
import validation

def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

test_logo = np.loadtxt("./test_logo copy.txt", dtype=float)

N = test_logo.shape[0]
x = test_logo.flatten(order="F")


A,_,_,_ = paralleltomo(N)

print(A.shape, x.shape, test_logo.shape)

b = A @ x

noise_b = add_noise_float(b,0,0.0001)

model_ridge = sklin.Ridge(alpha=40, fit_intercept=False)
model_ridge.fit(A, noise_b)
im_recov_ridge = np.reshape(model_ridge.coef_, (N,N), order = "F")

kmeans_image, _ = validation.kmean_clust(im_recov_ridge)

plt.imshow(test_logo, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(im_recov_ridge, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(kmeans_image, cmap="gray")
plt.axis("off")
plt.show()









