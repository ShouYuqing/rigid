"""
3D image augmentation
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

def rotate_img(img, vol_size, theta, beta ,omega):
    """
    3D image rotation in three axis
    if do not want rotate: value equals 360
    :param img: original image
    :param vol_size: grid's size
    :param theta: first dimension rotation
    :param beta: second dimension rotation
    :param omega: third dimension rotation
    :return: rotated image
    """
    # construct the transform matrix used for rotation
    theta = (theta / 180) * math.pi
    beta = (beta / 180) * math.pi
    omega = (omega / 180) * math.pi
    TX = np.array([[1, 0, 0, 0],
                  [0, math.cos(theta), math.sin(theta), 0],
                  [0, -math.sin(theta), math.cos(theta), 0],
                  [0, 0, 0, 1]])
    TY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                  [0, 1, 0, 1],
                  [math.sin(beta), 0, math.cos(beta), 0],
                  [0, 0, 0, 1]])
    TZ = np.array([[math.cos(omega), math.sin(omega), 0, 0],
                  [-math.sin(omega), math.cos(omega), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    T1 = np.array([[1, 0, 0, 0],
                  [0 ,1, 0, 0],
                  [0, 0, 1, 0],
                  [-int(vol_size[0]/2), -int(vol_size[1]/2), -int(vol_size[2]/2), 1]])
    T2 = np.array([[1, 0, 0, 0],
                  [0 ,1, 0, 0],
                  [0, 0, 1, 0],
                  [int(vol_size[0]/2), int(vol_size[1]/2), int(vol_size[2]/2), 1]])

    # construct grid
    x = np.arange(vol_size[1])
    y = np.arange(vol_size[0])
    z = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    for i in np.arange(vol_size[0]):
        for j in np.arange(vol_size[1]):
            for z in np.arange(vol_size[2]):
                coordinates = np.dot(T2, np.dot(TX, np.dot(T1, np.array([i, j, z, 1]).reshape(4, 1))))  # (4, 1)
                print(str(coordinates[0]) + ',' + str(coordinates[1]) + ',' + str(coordinates[2]))
                grid[i, j, z, 1] = coordinates[0]
                grid[i, j, z, 0] = coordinates[1]
                grid[i, j, z, 2] = coordinates[2]

    # interpolation
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    transformed_grid = np.stack((grid[:, :, :, 1], grid[:, :, :, 0], grid[:, :, :, 2]), 3)  # notice: the shifted_grid is reverse in x and y, so this step is used for making it back.
    post_img = interpn((yy, xx, zz), img[:, :, :], transformed_grid, method='nearest', bounds_error=False,
                       fill_value=0)
    return post_img

if __name__ == "__main__":
    img = np.load("atlas_norm.npz")
    img = img["vol"]
    print(img.shape)
    rotated_img = rotate_img(img, vol_size=(160,192,224), theta = 20, beta = 0, omega = 0)
    plt.figure()
    plt.imshow(rotated_img[80, :, :])
    plt.savefig("rotated.png")
    plt.figure()
    plt.imshow(img[80, :, :])
    plt.savefig("origin.png")
