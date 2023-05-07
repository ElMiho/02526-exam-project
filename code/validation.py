import numpy as np
import skimage as ski
from paralleltomo import paralleltomo
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import scipy.cluster
import binascii
from sklearn.cluster import KMeans
from PIL import Image

# from sklearn import KMeans

def pixel2pred(x, attenuations):
    if x <= attenuations[0] + 1e-10:
        idx = 0
    elif x <= attenuations[1] + 1e-10:
        idx = 1
    else:
        idx = 2

    return idx

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

def circle_to_im(vec):
    n = vec.shape[0]
    return_im = np.zeros((n,n))
    c = n//2
    for i in range(n):
        for j in range(n):
            if ((i - c)**2 + (j - c)**2) <= c**2:   #Area to evaluate
                return_im[i][j] = vec[i+j]
    return return_im

def predictions(recovered_image, original_image, attenuations):
    # Input: recovered image, original image, attenuations in the order wood, bismuth, steel
    # Returns a 3x3 confusion matrix and the predicted image based on most likelihood
    n,m = recovered_image.shape
    c = n//2
    radius = c - 1
    predicted_image = np.zeros((n,m))
    
    confusion_matrix = np.zeros((5,5), dtype=object)
    confusion_matrix[:,0] = ['-', 'Predicted', 'Wood', 'Bismuth', 'Steel']
    confusion_matrix[0,:] = ['-', 'Actual', 'Wood', 'Bismuth', 'Steel']
    confusion_matrix[:,1] = ['-','-','-','-','-']
    confusion_matrix[1,:] = ['-','-','-','-','-']

    for i in range(n):
        for j in range(m):
            if ((i - c)**2 + (j - c)**2) < radius**2:   #Area to evaluate
                x = recovered_image[i,j]
                y = np.abs([x-attenuations[0],x-attenuations[1],x-attenuations[2]]) #Calculate max-likelyhood
                idx = np.argmin(y)                          # Predicted class
                predicted_image[i,j] = attenuations[idx]    # Add value to image
                if(idx==0):     # Predicted Wood 
                    if(original_image[i,j]==attenuations[0]):   # Actual value wood
                        confusion_matrix[2,2] += 1              
                    elif(original_image[i,j]==attenuations[1]): # Actual value bismuth
                        confusion_matrix[2,3] += 1
                    elif(original_image[i,j]==attenuations[2]): # Actual value steel
                        confusion_matrix[2,4] += 1
                elif(idx==1):   # Predicted Bismuth 
                    if(original_image[i,j]==attenuations[0]):   # Actual value wood
                        confusion_matrix[3,2] += 1              
                    elif(original_image[i,j]==attenuations[1]): # Actual value bismuth
                        confusion_matrix[3,3] += 1
                    elif(original_image[i,j]==attenuations[2]): # Actual value steel
                        confusion_matrix[3,4] += 1
                elif(idx==2):   # Predicted Steel 
                    if(original_image[i,j]==attenuations[0]):   # Actual value wood
                        confusion_matrix[4,2] += 1              
                    elif(original_image[i,j]==attenuations[1]): # Actual value bismuth
                        confusion_matrix[4,3] += 1
                    elif(original_image[i,j]==attenuations[2]): # Actual value steel
                        confusion_matrix[4,4] += 1

    return predicted_image,confusion_matrix

def downsized_unique_value(resized_im,attenuations):
    # Input: resized image, attenuations in the order wood, bismuth, steel
    # Returns: The most likely value of the resized image
    n,m = resized_im.shape
    c = n//2
    radius = c-1
    im_unique_val = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if ((i - c)**2 + (j - c)**2) < radius**2:   #Area to evaluate
                x = resized_im[i,j]
                y = np.abs([x-attenuations[0],x-attenuations[1],x-attenuations[2]]) #Calculate max-likelyhood
                idx = np.argmin(y)                          # Predicted class
                im_unique_val[i,j] = attenuations[idx]    # Add value to image
    return im_unique_val

testImage = np.load("./testimage.npy")

unique_values = sorted(np.unique(testImage))

NUM_CLUSTERS = 4
# kmeans = KMeans(n_clusters=NUM_CLUSTERS)

downSamplingFactor = 100
N = int(testImage.shape[1]/downSamplingFactor)
resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor, func=np.mean)

# Convert the image to a numpy array
# image_array = resizedImage

print(f"Image shape: ", resizedImage.shape)
image_array = return_circle(resizedImage)

# Reshape the array to a 2D array of pixels
pixel_values = image_array.reshape((-1, 1))

# Perform k-means clustering with 4 clusters
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=628).fit(pixel_values)

# Get the labels and centroid values for each pixel
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Get the mean value in each cluster
cluster_values = []
for i in range(kmeans.n_clusters):
    mask = labels == i
    cluster_pixels = pixel_values[mask]
    cluster_mean = np.mean(cluster_pixels)
    cluster_values.append(cluster_mean)

print("Cluster values: ", sorted(cluster_values))

predicted_im = labels.reshape(50,50)
plt.imshow(predicted_im)
plt.show()


shape = testImage.shape
#testImage = testImage.reshape(scipy.product(shape[:2]), shape[2]).astype(float)



# print('finding clusters')
# codes, dist = scipy.cluster.vq.kmeans(resizedImage, NUM_CLUSTERS)
# print('cluster centres:\n', codes)
# print(f"Dist: {dist}")

# vecs, dist = scipy.cluster.vq.vq(resizedImage, codes)         # assign codes
# counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

# index_max = scipy.argmax(counts)                    # find most frequent
# peak = codes[index_max]
# colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
# print('most frequent is %s (#%s)' % (peak, colour))
# plt.show()

# # Wood, Iron, Bismuth
# print("Air: {} | Wood: {} | Iron: {} | Bismuth: {}".format(*unique_values))
# att_coefs = unique_values[1:]
# downSamplingFactor = 100

# N = int(testImage.shape[1]/downSamplingFactor)
# resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor, func=np.mean)

# x_input = resizedImage.flatten(order="F")

# A,_,_,_ = paralleltomo(N)

# b = A @ x_input

# im_recov, _, _, _ = np.linalg.lstsq(A, b)

# im_recov = np.reshape(im_recov, (N, N), order="F")

# im_original_unique = downsized_unique_value(resizedImage, att_coefs)

# predicted_im,confusion_matrix, = predictions(im_recov,im_original_unique,att_coefs)
# print(confusion_matrix)
# plt.figure("Im original unique")
# plt.imshow(im_original_unique)

# plt.figure("Im recov")
# plt.imshow(im_recov)

# plt.figure("Predicted im")
# plt.imshow(predicted_im)
# plt.show()
# print(np.unique(im_original_unique))