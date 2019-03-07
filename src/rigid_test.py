"""
test for rigid registration, from shift flow output a rigid matrix.
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


def test( iter_num, gpu_id, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16,3], model_name = "vm2_cc", sample_num = 10):
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
    x = np.linspace(0, 160-16, sample_num)
    x = x.astype(np.int32)
    y = np.linspace(0, 190-19, sample_num)
    y = y.astype(np.int32)
    z = np.linspace(0, 220-22, sample_num)
    z = z.astype(np.int32)
    index = np.rollaxis(np.array(np.meshgrid(y, x, z)), 0, 4)
    x = index[:, :, :, 0]
    y = index[:, :, :, 1]
    z = index[:, :, :, 2]
    print("x" + str(index[:, :, :, 0]))
    print("y" + str(index[:, :, :, 0]))
    print("z" + str(index[:, :, :, 0]))
    print("index's shape:"+str(index.shape))

    # Y in formula
    x_flow = np.arange(vol_size[0])
    y_flow = np.arange(vol_size[1])
    z_flow = np.arange(vol_size[2])
    grid = np.rollaxis(np.array((np.meshgrid(y_flow, x_flow, z_flow))), 0, 4)#original coordinate
    print("grid's shape:" + str(grid.shape))
    print("flow's shape:" + str(flow.shape))
    print("grid[:, :, :, 0]'s shape:" + str(grid[:, :, :, 0].shape))
    grid_x = grid_sample(x, y, z, grid[:, :, :, 0], sample_num)
    grid_y = grid_sample(x, y, z, grid[:, :, :, 1], sample_num)
    grid_z = grid_sample(x, y, z, grid[:, :, :, 2], sample_num)# (10,10,10)
    sample = flow + grid

    sample = sample[x, y, z]
    sample_x = sample[x, y, z, 0]
    sample_y = sample[x, y, z, 1]
    sample_z = sample[x, y, z, 2]#(10,10,10)

    sum_x = np.sum(flow[:, :, :, 0])
    sum_y = np.sum(flow[:, :, :, 1])
    sum_z = np.sum(flow[:, :, :, 2])

    ave_x = sum_x/(vol_size[0] * vol_size[1] * vol_size[2])
    ave_y = sum_y/(vol_size[0] * vol_size[1] * vol_size[2])
    ave_z = sum_z/(vol_size[0] * vol_size[1] * vol_size[2])

    # formula
    Y = np.zeros((10, 10, 10, 4))
    X = np.zeros((10, 10, 10, 4))
    T = np.array([ave_x, ave_y, ave_z, 1])#(4,1)
    R = np.zeros((10, 10, 10, 4, 4))

    for i in np.arange(10):
        for j in np.arange(10):
            for z in np.arange(10):
                Y[i, j, z, :] = np.array([sample_x[i,j,z], sample_y[i,j,z], sample_z[i,j,z], 1])

    for i in np.arange(10):
        for j in np.arange(10):
            for z in np.arange(10):
                X[i, j, z, :] = np.array([grid_x[i, j, z], grid_y[i, j, z], grid_z[i, j, z], 1])

    for i in np.arange(10):
        for j in np.arange(10):
            for z in np.arange(10):
                R[i, j, z, :] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X[i, j, z, :]), X[i, j, z, :])), X[i, j, z, :]), T)

    # multiply R with


def grid_sample(x, y, z, grid, sample_num):
    """
    sample the grid with x y z index grid
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


    flow_sample = interpn((x, y, z), X_seg[0, :, :, :, 0], flow, method = 'nearest', bounds_error = False, fill_value = 0)

    # sample the Y(Warped_X)
    # # (3,10,10,10) grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    # To do above

    # Warp segments with flow
    #flow = pred[1][0, :, :, :, :]
    #sample = flow+grid
    #sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    #warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

    #vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
    #print(np.mean(vals), np.std(vals))

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
