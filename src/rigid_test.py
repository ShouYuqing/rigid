"""
rigid registration
"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators
import neuron.layers as nrn_layers
import neuron.utils as util
import neuron.plot as nplt

def test(iter_num, gpu_id, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16,3], model_name = "vm2_cc", sample_num = 20, grid_dimension = 4):
    """
    Test of the rigid registration by calculating the dice score between the atlas's segmentation and warped image's segmentation
    :param iter_num: iteration number
    :param gpu_id: gpu id
    :param vol_size: volume's size
    :param nf_enc: number of encode
    :param nf_dec: number of decoder
    :param model_name: load model's name
    :param sample_num: sample grid's dimension, this can be changed to improve the performance
    :param grid_dimension: R(in the formula)'s dimension
    :return: None
    """
    gpu = '/gpu:' + str(gpu_id)

    # Anatomical labels we want to evaluate
    labels = sio.loadmat('../data/labels.mat')['labels'][0]

    atlas = np.load('../data/atlas_norm.npz')
    atlas_vol = atlas['vol']
    atlas_seg = atlas['seg']
    atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name + '.h5', by_name=True)

    X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')

    with tf.device(gpu):
        pred = net.predict([X_vol, atlas_vol])

    # get flow
    flow = pred[1][0, :, :, :, :]

    # Compute A(all about coordinate computation)
    x = np.linspace(0, (vol_size[0]/sample_num)*(sample_num-1), sample_num)
    x = x.astype(np.int32)
    y = np.linspace(0, (vol_size[1]/sample_num)*(sample_num-1), sample_num)
    y = y.astype(np.int32)
    z = np.linspace(0, (vol_size[2]/sample_num)*(sample_num-1), sample_num)
    z = z.astype(np.int32)
    index = np.rollaxis(np.array(np.meshgrid(y, x, z)), 0, 4)
    print("index's shape:" + str(index.shape))
    print(index[:, :, :, 1])
    print(index[:, :, :, 0])
    print(index[:, :, :, 2])
    print("vol_size[0]'s shape:" + str(vol_size[0]))
    print(x)
    print(y)
    print(z)
    print("vol_size[1]'s shape:" + str(vol_size[1]))
    print("vol_size[2]'s shape:" + str(vol_size[2]))
    x = index[:, :, :, 1]
    y = index[:, :, :, 0]
    z = index[:, :, :, 2]
    print("x's shape:" + str(x.shape))
    print("y's shape:" + str(y.shape))
    print("z's shape:" + str(z.shape))

    # Y in formula
    x_flow = np.arange(vol_size[0])
    y_flow = np.arange(vol_size[1])
    z_flow = np.arange(vol_size[2])
    grid = np.rollaxis(np.array((np.meshgrid(y_flow, x_flow, z_flow))), 0, 4)#original coordinate
    print("grid's shape:" + str(grid.shape))
    print("grid_sample 0:" + str(grid[:, :, :, 0].shape))
    grid_x = grid_sample(x, y, z, grid[:, :, :, 1], sample_num)
    grid_y = grid_sample(x, y, z, grid[:, :, :, 0], sample_num)
    grid_z = grid_sample(x, y, z, grid[:, :, :, 2], sample_num)#X (10,10,10)

    sample = flow + grid
    sample_x = grid_sample(x, y, z, sample[:, :, :, 1], sample_num)
    sample_y = grid_sample(x, y, z, sample[:, :, :, 0], sample_num)
    sample_z = grid_sample(x, y, z, sample[:, :, :, 2], sample_num)#Y (10,10,10)

    sum_x = np.sum(flow[:, :, :, 1])
    print(flow[:, :, :, 1])
    print(flow[:, :, :, 0])
    print(flow[:, :, :, 2])

    sum_y = np.sum(flow[:, :, :, 0])
    sum_z = np.sum(flow[:, :, :, 2])

    ave_x = sum_x/(vol_size[0] * vol_size[1] * vol_size[2])
    ave_y = sum_y/(vol_size[0] * vol_size[1] * vol_size[2])
    ave_z = sum_z/(vol_size[0] * vol_size[1] * vol_size[2])

    # formula
    Y = np.zeros((sample_num, sample_num, sample_num, grid_dimension))
    X = np.zeros((sample_num, sample_num, sample_num, grid_dimension))
    T = np.array([ave_x, ave_y, ave_z, 1])#(4,1)
    #R = np.zeros((10, 10, 10, grid_dimension, grid_dimension))

    for i in np.arange(sample_num):
        for j in np.arange(sample_num):
            for z in np.arange(sample_num):
                Y[i, j, z, :] = np.array([sample_x[i,j,z], sample_y[i,j,z], sample_z[i,j,z], 1])

    for i in np.arange(sample_num):
        for j in np.arange(sample_num):
            for z in np.arange(sample_num):
                X[i, j, z, :] = np.array([grid_x[i, j, z], grid_y[i, j, z], grid_z[i, j, z], 1])

    X = X.reshape((sample_num * sample_num * sample_num, grid_dimension))
    Y = Y.reshape((sample_num * sample_num * sample_num, grid_dimension))
    R = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)# R

    # build new grid(Use R to do the spatial transform)
    shifted_x = np.arange(vol_size[0])
    shifted_y = np.arange(vol_size[1])
    shifted_z = np.arange(vol_size[2])
    shifted_grid = np.rollaxis(np.array((np.meshgrid(shifted_y, shifted_x, shifted_z))), 0, 4)
    print(shifted_grid.shape)
    for i in np.arange(vol_size[0]):
        for j in np.arange(vol_size[1]):
            for z in np.arange(vol_size[2]):
                coordinates = np.dot(R, np.array([i, j, z, 1]).reshape(4,1)) +  T.reshape(4,1)
                #print("voxel." + '(' + str(i) + ',' + str(j) + ',' + str(z) + ')')
                shifted_grid[i, j, z, 1] = coordinates[0]
                shifted_grid[i, j, z, 0] = coordinates[1]
                shifted_grid[i, j, z, 2] = coordinates[2]

    # interpolation
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    shifted_grid = np.stack((shifted_grid[:, :, :, 1], shifted_grid[:, :, :, 0], shifted_grid[:, :, :, 2]), 3)# notice: the shifted_grid is reverser in x and y, so this step is used for making it back.
    warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], shifted_grid, method='nearest', bounds_error=False, fill_value=0)# rigid registration

    # CVPR
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    sample = flow + grid
    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_seg2 = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)# deformable registration

    # compute dice
    vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
    vals2, _ = dice(X_seg[0, :, :, :, 0], atlas_seg, labels=labels, nargout=2)
    vals3, _ = dice(warp_seg2, atlas_seg, labels=labels, nargout=2)
    print("dice before:")
    print(np.mean(vals2), np.std(vals2))
    print("dice after deformable registration:")
    print(np.mean(vals3), np.std(vals3))
    print("dice after rigid registration:")
    print(np.mean(vals), np.std(vals))

    # plot
    fig1, axs1 = nplt.slices(warp_seg[100, :, :], do_colorbars=True)
    fig1.savefig('warp_seg100.png')
    fig2, axs2 = nplt.slices(warp_seg[130, :, :], do_colorbars=True)
    fig2.savefig('warp_seg130.png')
    fig3, axs3 = nplt.slices(atlas_seg[100, :, :], do_colorbars=True)
    fig3.savefig('atlas_seg100.png')
    fig4, axs4 = nplt.slices(atlas_seg[130, :, :], do_colorbars=True)
    fig4.savefig('atlas_seg130.png')

    #print(R)
    #print(T)
    #print("point (1, 1, 1) after registration:")
    #print(np.dot(R, np.ones((4,1))) +  T.reshape(4,1))




def grid_sample(x, y, z, grid, sample_num):
    """
    sample the grid with x y z index
    :param x: x index grid
    :param y: y index grid
    :param z: z index grid
    :param grid: grid to be sampled
    :param sample_num: sample num, then sample the grid with num*num*num
    :return: grid after sample
    """
    sampled_grid = np.ones((sample_num, sample_num, sample_num))
    for i in np.arange(sample_num):
        for j in np.arange(sample_num):
            for m in np.arange(sample_num):
                sampled_grid[i, j, m] = grid[x[i, j, m], y[i, j, m], z[i, j, m]]
    return sampled_grid

    #sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    #warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

    # sample the X(add coordinates)

    #flow_sample = interpn((x, y, z), X_seg[0, :, :, :, 0], flow, method = 'nearest', bounds_error = False, fill_value = 0)

    # sample the Y(Warped_X)
    # # (3,10,10,10) grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
