"""
3D image augmentation
"""
import math
import numpy as np

def rotate_img(img, vol_size, theta, beta ,omega):
    """
    3D image rotation in three axis
    if do not want rotate: value equals 360
    :param img: original image
    :param vol_size: grid's size
    :param theta: x
    :param beta: y
    :param omega: z
    :return: rotated image
    """
    # construct the transform matrix used for rotation
    theta = (theta / 180) * math.pi
    beta = (beta / 180) * math.pi
    omega = (omega / 180) * math.pi
    TX = np.array([[1, 0, 0, 0],
                  [0, math.cos(beta), math.sin(beta), 0],
                  [0, -math.sin(beta), math.cos(beta), 0],
                  [0, 0, 0, 1]])
    TY = np.array([[],
                  [],
                  [],
                  []])
    TZ = np.array([[],
                  [],
                  [],
                  []])

    T1 = np.array([[],
                  [],
                  [],
                  []])
    T2 = np.array([[],
                  [],
                  [],
                  []])

    # construct grid
    x = np.arange(vol_size[1])
    y = np.arange(vol_size[0])
    z = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    for i in np.arange(vol_size[0]):
        for j in np.arange(vol_size[1]):
            for z in np.arange(vol_size[2]):
                grid[i, j, z, 0] =
                grid[i, j, z, 1] =
                grid[i, j, z, 2] =


    # interpolation


    # construct return grid

    return img
