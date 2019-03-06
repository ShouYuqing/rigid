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


def test( iter_num, gpu_id, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16,3], model_name = "vm2.cc", sample_num = 10):
    gpu = '/gpu:' + str(gpu_id)

    # Anatomical labels we want to evaluate
    labels = sio.loadmat('../data/labels.mat')['labels'][0]

    atlas = np.load('../data/atlas_norm.npz')
    atlas_vol = atlas['vol']
    atlas_seg = atlas['seg']
    atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name +
                             '/' + str(iter_num) + '.h5')

    X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')

    with tf.device(gpu):
        pred = net.predict([X_vol, atlas_vol])

    # get flow
    flow = pred[1][0, :, :, :, :]

    # Compute A(all about coordinate computation)
    x_flow = np.arange(vol_size[0])
    y_flow = np.arange(vol_size[1])
    z_flow = np.arange(vol_size[2])
    grid = np.array((np.meshgrid(x_flow, y_flow, z_flow)))# grid: original coordinate

    # Y in formula
    sample = flow + grid# sample: coordinate after shifting, size:3*(...), each tensor stores the coordinates for voxel








    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

    # sample the X(add coordinates)
    x = linspace(0, vol_size[0], sample_num + 1)
    y = linspace(0, vol_size[1], sample_num + 1)
    z = linspace(0, vol_size[2], sample_num + 1)  # sample-coordinate

    flow_sample = interpn((x, y, z), X_seg[0, :, :, :, 0], flow, method = 'nearest', bounds_error = False, fill_value = 0)

    # sample the Y(Warped_X)
    # # (3,10,10,10) grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    # To do above

    # Warp segments with flow
    flow = pred[1][0, :, :, :, :]
    sample = flow+grid
    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

    vals, _ = dice(warp_seg, atlas_seg, labels=labels, nargout=2)
    print(np.mean(vals), np.std(vals))

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3])
