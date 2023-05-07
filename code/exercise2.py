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

def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

def add_noise_percent(b, mean_percent=50, std_percent=70):
    return add_noise_float(b, np.mean(b)*mean_percent/100, np.mean(b)*std_percent/100)

def threshold(image, color='gray'):
    '''Expects image to be a numpy array, shape: (N,N)
    Returns a numpy array, shape: (N,N) if color == 'gray'
    Returns a numpy array, shape: (N,N,3) if color == 'rgb'''
    '''Color map is either rgb or gray'''
    truncations = [0, 157, 174, 440] #Estimate of the mean values of the colors (in grayscale) (This process can be automated)
    mean_truncation = [(truncations[i]+truncations[i+1])/2 for i in range(len(truncations)-1)]
    index = 0 
    if color == 'rgb':
        mean_truncation = [i/255 for i in mean_truncation]
        imageRGB = np.zeros((image.shape[0], image.shape[1], 3))
        colors = [[0,255,255], [255,0,0], [0,255,0], [255,255,0]] #cyan, red, green, yellow
        for i in range(0,imageRGB.shape[2]):
            imageRGB[:,:,i] = image[:,:]/255
        pixels = []
           
        for i in range(len(mean_truncation)-1):
            pixels.append(np.where((imageRGB > mean_truncation[i]) & (imageRGB < mean_truncation[i+1])))
            print(mean_truncation[i], mean_truncation[i+1])

        #edge cases
        pixels.append(np.where(imageRGB < mean_truncation[0]))
        pixels.append(np.where(imageRGB > mean_truncation[-1]))
        for pixel in pixels:
            imageRGB[pixel[0],pixel[1],:] = colors[index]
            index+=1
        return imageRGB

    elif color == 'gray':
        colors = [100,170,0,255]
        pixels = []
        for i in range(len(mean_truncation)-1):
            pixels.append(np.where((image > mean_truncation[i]) & (image < mean_truncation[i+1])))
        #edge cases
        pixels.append(np.where(image < mean_truncation[0]))
        pixels.append(np.where(image > mean_truncation[-1]))

        for pixel in pixels:
            image[pixel[0],pixel[1]] = colors[index]
            index+=1

    return image

def kmean_clust(resized_im):
    image_array = return_circle(resized_im)

    # Reshape the array to a 2D array of pixels
    pixel_values = image_array.reshape((-1, 1))

    # Perform k-means clustering with 4 clusters
    NUM_CLUSTERS = 4
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=628).fit(pixel_values)

    # Get the labels and centroid values for each pixel
    labels = kmeans.labels_

    # Get the mean value in each cluster
    cluster_values = []
    for i in range(kmeans.n_clusters):
        mask = labels == i
        cluster_pixels = pixel_values[mask]
        cluster_mean = np.mean(cluster_pixels)
        cluster_values.append(cluster_mean)

    cluster_values = sorted(cluster_values)
    att_coefs = cluster_values[1:]
    
    return labels.reshape(N,N), att_coefs

def return_circle(im):
    n,m = im.shape
    return_im = np.zeros((n,m))
    c = n//2
    radius = c - 1
    for i in range(n):
        for j in range(m):
            if ((i - c)**2 + (j - c)**2) < radius**2:   #Area to evaluate
                return_im[i,j] = im[i,j]
    return return_im


testImage = np.load('testImage.npy')

max_pixel = np.max(testImage)
print(f"max pixel {max_pixel}\n")
testImage = testImage / max_pixel

downSamplingFactor = 100

N = 50
resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor)
x = resizedImage.flatten(order="F")


A,theta,p,d = paralleltomo(N)

print(resizedImage.shape, A.shape, x.shape)

b = A @ x

# print(b)

#print([(i,b) for i,b in enumerate(np.unique(np.round(b, 4))) if b > 0.0001])

picture1, _, _, _ = linalg.lstsq(A, b)
print("done")

print(picture1, len(picture1))
print(type(picture1), type(picture1[0]))
picture1 = np.reshape(picture1, (50,50), order="F")

# ## Add noise
# mean = np.mean(b)/100
# std = np.std(b)/1000
# print(f"mean : {mean} std: {std}\n")
# noise = np.random.normal(loc=mean, scale=std, size=b.shape)

# print(f"no noise: {b[:50]} noise: {noise[:50]}")
# noise_b = add_noise_float(b, mean, std)

# noise_picture, _, _, _ = linalg.lstsq(A, noise_b)
# noise_picture = np.reshape(noise_picture, (50, 50), order="F")

# plt.figure(1)
# # print(testImage.shape)
# plt.subplot(2, 2, 1)
# plt.title("Originale image")
# plt.imshow(testImage, cmap="gray")

# plt.subplot(2, 2, 2)
# plt.title("Resized image")
# plt.imshow(resizedImage, cmap="gray")

# plt.subplot(2, 2, 3)
# plt.title("Reconstructed - basic")
# plt.imshow(picture1)

# plt.subplot(2, 2, 4)
# plt.title("Reconstructed - noise")
# plt.imshow(noise_picture)

# plt.show()

# std_percentages = [0.01, 0.1, 1, 10]

## Plot of different means
# plt.figure(2)

std_percentages = list(range(0,9,2))
thetas = [
    np.matrix(np.linspace(1 + 0, 1 + 180-step_size, 180 // step_size)) for step_size in range(1,6)
]

down_sampling_factors = [
    200, 150, 100
]

n_alphas = 10
# alphas = np.logspace(3, -2, n_alphas)
alphas = [1/2, 1, 10]

for a in alphas:
    index = 1
    for down_scaling_factor in tqdm(down_sampling_factors):
        resizedImage = ski.measure.block_reduce(testImage, block_size=down_scaling_factor, func=np.mean)
        N, _N = resizedImage.shape

        orignal_image_unique, att_coefs = kmean_clust(resizedImage)

        d_vals = [
            np.sqrt(2) * N
        ]
        p_vals = [
            int(np.sqrt(2) * N/(0.5*i)) for i in range(2, 8)
        ]

        for d_val in d_vals:
            for p_val in p_vals:
                for theta in thetas:
                    x = resizedImage.flatten(order="F")
                    
                    A, _,_ ,_ = paralleltomo(N, theta, p_val, d_val)
                    condA = np.linalg.cond(A)

                    if condA < 1000:
                        b = A @ x
                        
                        plot_dim = len(std_percentages)**0.5
                        n_rows = int(np.ceil(plot_dim))
                        n_cols = int(plot_dim)
                        start_time = time.time()
                        
                        plt.figure(index)
                        plt.tight_layout()
                        index += 1
                        for i, std_p in enumerate(std_percentages):
                            plt.subplot(n_rows, n_cols, i+1)
                            plt.title(f"std_p: {std_p}")

                            noise_b = add_noise_percent(b, 0, std_p)
                            model_ridge = sklin.Ridge(alpha=a, fit_intercept=False)
                            model_ridge.fit(A, noise_b)

                            # model_lasso = sklin.Lasso(alpha=a)
                            # model_lasso.fit(A,noise_b)

                            im_recov_ridge = np.reshape(model_ridge.coef_,(N,N), order = "F")
                            # im_recov_lasso = np.reshape(model_lasso.coef_,(N,N), order = "F")

                            predicted_image = kmean_clust(im_recov_ridge)


                            
                            plt.imshow(im_recov_ridge)
                        
                        folder = "images"
                        print(f"Time taken for plotting: {time.time() - start_time:.4f}s")
                        plt.suptitle(f"N={N}, cond={condA}, theta_shape={theta.shape}, p={p_val}, d={round(d_val,4)}, down_scaling_factor={down_scaling_factor}, index={index}", fontsize=12)
                        plt.savefig(f"./{folder}/N={N}, cond={condA}, theta_shape={theta.shape}, p={p_val}, d={round(d_val,4)}, down_scaling_factor={down_scaling_factor}, alpha={a}, time={time.time() - start_time}, index={index}.png")

                    break
                break
            break
        break
    break

# for a in alphas:
#     model_ridge = sklin.Ridge(alpha=a, fit_intercept=False)
#     model_ridge.fit(A, b)

#     plt.figure(a)
#     im_recov_ridge = np.reshape(model_ridge.coef_,(N,N), order = "F")
#     plt.imshow(im_recov_ridge)

# plt.show()