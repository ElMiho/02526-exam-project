import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import skimage as ski
from paralleltomo import paralleltomo
import sklearn.linear_model as sklin
from generate_test_functions import *

def generate_confusion(predicts,original_im,im_pred,att_coef_pellet,att_coef_wood):
    
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    
    for i,j in predicts:
        if(im_pred[i,j]==att_coef_pellet):   #Predicted pellet
            if(original_im[i,j]==im_pred[i,j]):    #Pellet prediction true
                true_pos += 1
            else: # Pellet prediction false
                false_pos += 1
        elif(original_im[i,j]==att_coef_wood):   #Predicted wood
            if(original_im[i,j]==im_pred[i,j]): #Wood prediction true
                true_neg += 1
            else:   #Wood prediction false
                false_neg += 1
    confusion_matrix = np.array([[true_pos, false_pos], [false_neg, true_neg]])
    return confusion_matrix

def predictions(recov_im, att):
    # Attenuations, wood, bismuth, steel
    n,m = recov_im.shape
    c = n//2
    pred_bismuth = []
    pred_steel = []
    pred_wood = []
    pred_image = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if ((i - c)**2 + (j - c)**2) <= c**2:
                x = recov_im[i,j]
                y = np.abs([x-att[0],x-att[1],x-att[2]])
                idx = np.argmin(y)
                pred_image[i,j] = att[idx]
                if(idx==1):
                    pred_bismuth.append((i,j))
                elif(idx==2):
                    pred_steel.append((i,j))
                else:
                    pred_wood.append((i,j))
    return pred_image,pred_bismuth,pred_steel,pred_wood

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

testImage = np.load("./testimage.npy")

#max_pixel = np.max(testImage)
#testImage = testImage / max_pixel

unique_values = np.unique(testImage)
# Wood, Bismuth, Iron
att_wood = unique_values[1]
att_bismuth = unique_values[2]
att_iron = unique_values[3]

downSamplingFactor = 100

N = int(testImage.shape[1]/downSamplingFactor)
resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor)
x_input = resizedImage.flatten(order="F")

A,_,_,_ = paralleltomo(N)

b = A @ x_input

im_recov, _, _, _ = np.linalg.lstsq(A, b)

im_recov = np.reshape(im_recov, (N, N), order="F")


predicted_im,predicted_bismuth,predicted_steel = predictions(im_recov,unique_values[1:])

confusion_steel = generate_confusion(predicted_steel,im_recov,predicted_im,att_steel,att_wood)
confusion_bismuth = generate_confusion(predicted_bismuth,im_recov,predicted_im,att_bismuth,att_wood)

print(f"confusion matrix steel:\n {confusion_steel}")
print(f"confusion matrix bismuth:\n {confusion_bismuth}")

# n = 50
# num_pellets = 100
# pellet_size = 1

# # Wood, Bismuth, Iron at 16.4 keV
# att_coefs = [1.0646,147.8,48.3665]
# #im = generate_im(n,att_coefs,num_pellets,pellet_size)

# x_input = im.flatten(order = "F")

# theta = np.matrix(np.linspace(1 + 0, 1 + 180-2, 180 // 2))
# p=30
# A, _,_ ,_ = paralleltomo(n,theta,p)
# print(f"Condition number {np.linalg.cond(A)}")

# b = A @ x_input

# b_poisson = poisson_noise(b,lamb = 1)


# # Plot of pixels as a function of alpha ||Ax-b||_2 + alpha*||x||_2
# n_alphas = 10
# alphas = np.logspace(10, -2, n_alphas)
# no_noise_predict = []
# noise_predict = []
# for a in alphas:
#     model = sklin.Ridge(alpha=a, fit_intercept=False)

#     model.fit(A, b)
#     im_recov_no_noise = np.reshape(model.coef_,(n,n))

#     model.fit(A,b_poisson)
#     im_recov_noise = np.reshape(model.coef_,(n,n))

#     no_noise_predict.append(likelihood(im_recov_no_noise,att_coefs))
#     noise_predict.append(likelihood(im_recov_noise,att_coefs))

# plt.figure(1)
# # Plot images
# for i in range(n_alphas):
#     plt.subplot(4,3,i+1)
#     plt.imshow(np.asarray(no_noise_predict[i]))

# plt.savefig(";))")
# plt.close()

# plt.figure(2)
# for i in range(n_alphas):
#     plt.subplot(4,3,i+1)
#     plt.imshow(np.asarray(noise_predict[i]))

# plt.savefig(";)))")
# plt.close()