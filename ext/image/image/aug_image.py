"""
3D operation&plot lib
"""

# lib
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

# plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def rotate_img(img, vol_size = (160,192,224), theta = 0, beta = 0 ,omega = 0):
    """
    3D image rotation in three axis
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
                  [0, math.cos(theta), math.sin(theta), 0],
                  [0, -math.sin(theta), math.cos(theta), 0],
                  [0, 0, 0, 1]])
    TY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                   [0, 1, 0, 0],
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

    # rotated grid
    for i in np.arange(vol_size[0]):
        for j in np.arange(vol_size[1]):
            for z in np.arange(vol_size[2]):
                coordinates = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.array([grid[i, j, z, 1], grid[i, j, z, 0], grid[i, j, z, 2], 1]).reshape(1, 4), T1), TX), T2)
                                                                        , T1), TY), T2), T1), TZ), T2) # (1, 4)
                grid[i, j, z, 1] = coordinates[0, 0]
                grid[i, j, z, 0] = coordinates[0, 1]
                grid[i, j, z, 2] = coordinates[0, 2]

    # interpolation
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    transformed_grid = np.stack((grid[:, :, :, 1], grid[:, :, :, 0], grid[:, :, :, 2]), 3)  # notice: the grid is reverse in x and y, so this step is used for making it back.
    post_img = interpn((yy, xx, zz), img[:, :, :], transformed_grid, method='nearest', bounds_error=False,
                   fill_value=0)
    return post_img


def rotate_img_p(img, vol_size = (160,192,224), theta = 0, beta = 0 ,omega = 0):
    """
    3D image rotation in three axis with parallel computing by cuda
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
                  [0, math.cos(theta), math.sin(theta), 0],
                  [0, -math.sin(theta), math.cos(theta), 0],
                  [0, 0, 0, 1]])
    TY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                   [0, 1, 0, 0],
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

    # rotated grid
    for i in np.arange(vol_size[0]):
        for j in np.arange(vol_size[1]):
            for z in np.arange(vol_size[2]):
                coordinates = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.array([grid[i, j, z, 1], grid[i, j, z, 0], grid[i, j, z, 2], 1]).reshape(1, 4), T1), TX), T2)
                                                                        , T1), TY), T2), T1), TZ), T2) # (1, 4)
                grid[i, j, z, 1] = coordinates[0, 0]
                grid[i, j, z, 0] = coordinates[0, 1]
                grid[i, j, z, 2] = coordinates[0, 2]

    # interpolation
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    transformed_grid = np.stack((grid[:, :, :, 1], grid[:, :, :, 0], grid[:, :, :, 2]), 3)  # notice: the grid is reverse in x and y, so this step is used for making it back.
    post_img = interpn((yy, xx, zz), img[:, :, :], transformed_grid, method='nearest', bounds_error=False,
                   fill_value=0)
    return post_img

def plot_grid(X, Y, Z):
    """
    plot for grid data, used for plot the registration field
    :param X: X grid
    :param Y: Y grid
    :param Z: Z grid
    :return: None
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-2, 2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    # test for rotate_img function
    #img = np.load("atlas_norm.npz")
    #img = img["vol"]
    #rotated_img = rotate_img(img, vol_size=(160,192,224), beta = 20)
    #plt.figure()
    #plt.imshow(rotated_img[:, 90, :])
    #plt.savefig("rotated.png")
    #plt.figure()
    #plt.imshow(img[:, 90, :])
    #plt.savefig("origin.png")

    # test for plot_grid function
    img = np.load("atlas_norm.npz")
    img = img["vol"]
    rotated_img = rotate_img(img, vol_size=(160, 192, 224), beta=5)

