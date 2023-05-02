from paralleltomo import paralleltomo
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import skimage as ski
from scipy import linalg
from tqdm import tqdm
import time
import cvxpy as cp

def add_noise_float(b, mean_noise, std_noise):
    return b + np.random.normal(loc=mean_noise, scale=std_noise, size=b.shape)

def add_noise_percent(b, mean_percent=50, std_percent=70):
    return add_noise_float(b, np.mean(b)*mean_percent/100, np.mean(b)*std_percent/100)

testImage = np.load('testImage.npy')

max_pixel = np.max(testImage)
print(f"max pixel {max_pixel}\n")
testImage = testImage / max_pixel

downSamplingFactor = 100

N = 50
resizedImage = ski.measure.block_reduce(testImage, block_size=downSamplingFactor)
x = resizedImage.flatten(order="F")

# A,theta,p,d = paralleltomo(N)

# print(resizedImage.shape, A.shape, x.shape)

# b = A @ x

# n = A.shape[1]
# x = cp.Variable(n)
# cost = cp.sum_squares(A @ x - b)
# prob = cp.Problem(cp.Minimize(cost))
# prob.solve()

# print(f"optimal value: {prob.value}")
# print(f"optimal x: {x.value}")
# print(f"max of x: {max(x.value)}")
# print(f"shape: {x.value.shape}")

# picture = np.asarray([x.value]).transpose()
# picture = np.reshape(picture, (N, N), order="F")
# print(f"new shape: {picture.shape}")
# plt.imshow(picture)
# plt.show()

std_percentages = list(range(0,9))
thetas = [
    np.matrix(np.linspace(1 + 0, 1 + 180-step_size, 180 // step_size)) for step_size in range(1,6)
]

down_sampling_factors = [
    200, 150, 100, 50
]

index = 1
for down_scaling_factor in tqdm(down_sampling_factors):
    resizedImage = ski.measure.block_reduce(testImage, block_size=down_scaling_factor)
    N, _N = resizedImage.shape

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

                    n = A.shape[1]
                    x = cp.Variable(n)
                    cost = cp.sum_squares(A @ x - noise_b)
                    prob = cp.Problem(cp.Minimize(cost))
                    prob.solve()

                    # noise_picture, _, _, _ = linalg.lstsq(A, noise_b)

                    noise_picture = np.asarray([x.value]).transpose()
                    noise_picture = np.reshape(noise_picture, (N, N), order="F")

                    # noise_picture = np.reshape(noise_picture, (N, N), order="F")
                    plt.imshow(noise_picture)
                
                print(f"Time taken for plotting: {time.time() - start_time:.4f}s")
                # plt.suptitle(f"N={N}, cond={condA}, theta_shape={theta.shape}, p={p_val}, d={round(d_val,4)}, down_scaling_factor={down_scaling_factor}, index={index}", fontsize=12)
                plt.savefig(f"./images-cvxpy/N={N}, cond={condA}, theta_shape={theta.shape}, p={p_val}, d={round(d_val,4)}, down_scaling_factor={down_scaling_factor}, index={index}.png")
                
    #             break
    #         break
    #     break
    # break



