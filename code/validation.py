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
    radius = c - 5
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

    return confusion_matrix

def downsized_unique_value(resized_im,attenuations, as_idx=False):
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
                if as_idx:
                    im_unique_val[i,j] = idx
    return im_unique_val


def calculate_conf_matrix(recovered_image, original_image, N):

    image_array = return_circle(recovered_image)

    # Reshape the array to a 2D array of pixels
    pixel_values = image_array.reshape((-1, 1))

    # Perform k-means clustering with 4 clusters
    NUM_CLUSTERS = 4
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

    cluster_values = sorted(cluster_values)
    att_coefs = cluster_values[1:]
    
    predicted_im = labels.reshape(N,N)
    
    #im_original_unique = downsized_unique_value(original_image, att_coefs)

    #confusion_matrix = predictions(predicted_im,im_original_unique,att_coefs)

    return predicted_im

def kmean_clust(resized_im):
    image_array = return_circle(resized_im)
    N = image_array.shape[0]

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

def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

# def add_noise_percent(b, mean_percent=50, std_percent=70):
#     return add_noise_float(b, np.mean(b)*mean_percent/100, np.mean(b)*std_percent/100)

if __name__ == "__main__":
    testImage = np.load("./testimage.npy")

    downSamplingFactor = 100
    N = int(testImage.shape[1]/downSamplingFactor)
    resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor, func=np.mean)

    orignal_image_unique, att_coefs_org = kmean_clust(resizedImage)

    x = resizedImage.flatten(order="F")

    A, _, _, _ = paralleltomo(N)
    
    b = A@x

    noise_b = add_noise_float(b, 0, 0.0001)
    x_recov,_,_,_ = np.linalg.lstsq(A, noise_b)
    
    
    recov_im = x_recov.reshape((N,N), order="F")

    recov_predict,att_recov = kmean_clust(recov_im)

    recov_predict_unique = downsized_unique_value(recov_predict,att_recov)
    att_coef_recov = np.unique(recov_predict_unique)
    
    print(f"Att original{att_coefs_org}\n")
    print(f"Att recovered{att_recov}\n")
    print(f"Att recovered_unique{att_coef_recov}")


    #print(confusion_matrix)
    plt.figure("Im original unique")
    plt.imshow(orignal_image_unique)

    plt.figure("Im recovered k means")
    plt.imshow(recov_predict)

    plt.figure("Im recovered unique")
    plt.imshow(recov_predict_unique)

    plt.show()





    # Convert the image to a numpy array
    # print(f"Image shape: ", resizedImage.shape)
    # image_array = return_circle(resizedImage)

    # # Reshape the array to a 2D array of pixels
    # pixel_values = image_array.reshape((-1, 1))

    # # Perform k-means clustering with 4 clusters --> Air, Wood, Iron, Bismuth
    # NUM_CLUSTERS = 4
    # kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=628).fit(pixel_values)

    # # Get the labels and centroid values for each pixel
    # labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    # # Get the mean value in each cluster
    # cluster_values = []
    # for i in range(kmeans.n_clusters):
    #     mask = labels == i
    #     cluster_pixels = pixel_values[mask]
    #     cluster_mean = np.mean(cluster_pixels)
    #     cluster_values.append(cluster_mean)

    # print("Cluster values: ", sorted(cluster_values))
    # att_coefs = sorted(cluster_values)[1:]

    # predicted_im = labels.reshape(N,N)
    
    # im_original_unique = downsized_unique_value(resizedImage, att_coefs)

    # confusion_matrix = predictions(predicted_im,im_original_unique,att_coefs)
