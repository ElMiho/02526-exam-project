# Imports
import matplotlib.pyplot as plt
import numpy as np
from paralleltomo import paralleltomo
import sklearn.linear_model as sklin
import skimage as ski
import validation

# Helper function
def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

# Your log with pellets :)
test_image = np.load('testImage.npy')
# Resize using mean block and 
test_image = ski.measure.block_reduce(test_image, block_size=50, func=np.mean)

# Helper variable - resolution
N = test_image.shape[0]

# Get image and projection matrix
x = test_image.flatten(order="F")
A,_,_,_ = paralleltomo(N)

# Print some stats
print(f"A.shape: {A.shape}, x.shape: {x.shape}, test_image.shape, {test_image.shape}")

# Make projection
b = A @ x

# Add noise (the default noise is pretty low, but so are the values in the given test image)
noise_b = add_noise_float(b, 0, 0.0001)

# Solve using ridge regression - choose another alpha if you dare ;)
model_ridge = sklin.Ridge(alpha=40, fit_intercept=False)
model_ridge.fit(A, noise_b)
im_recov_ridge = np.reshape(model_ridge.coef_, (N,N), order = "F")

# Use our K-means function to predict the four clusters: air, wood, iron and bismuth
kmeans_image, _ = validation.kmean_clust(im_recov_ridge)

# Save images
plt.figure(1)
plt.imshow(test_image)
plt.axis("off")
plt.savefig("test_image_628.png")

plt.figure(2)
plt.imshow(im_recov_ridge)
plt.axis("off")
plt.savefig("test_image_recov_628.png")

plt.figure(3)
plt.imshow(kmeans_image)
plt.axis("off")
plt.savefig("test_image_kmeans_628.png")









