"""
Use miccai model to do the rigid registration
"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
# import util
from medipy.metrics import dice
import datagenerators

# Test file and anatomical labels we want to evaluate
test_brain_file = open('.../data/test_examples.txt')
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

# atlas files
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
atlas_seg = atlas['seg']


def test(gpu_id, model_dir, iter_num,
         compute_type='GPU',  # GPU or CPU
         vol_size=(160, 192, 224),
         nf_enc=[16, 32, 32, 32],
         nf_dec=[32, 32, 32, 32, 16, 3],
         save_file=None):
    """
    test via segmetnation propagation
    works by iterating over some iamge files, registering them to atlas,
    propagating the warps, then computing Dice with atlas segmentations
    """

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # if testing miccai run, should be xy indexing.
        net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=True, indexing='xy')
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size)

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)

    # prepare a matrix of dice values
    dice_vals = np.zeros((len(good_labels), n_batches))
    for k in range(n_batches):
        # get data
        vol_name, seg_name = test_brain_strings[k].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        # predict transform
        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol])

        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[0, :, :, :, :]
            warp_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            # Rigid registration only by GPU
            flow = pred[0, :, :, :, :]
            # Compute A(all about coordinate computation)
            x = np.linspace(0, 160 - 16, sample_num)
            x = x.astype(np.int32)
            y = np.linspace(0, 190 - 19, sample_num)
            y = y.astype(np.int32)
            z = np.linspace(0, 220 - 22, sample_num)
            z = z.astype(np.int32)
            index = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
            x = index[:, :, :, 0]
            y = index[:, :, :, 1]
            z = index[:, :, :, 2]

            # Y in formula
            x_flow = np.arange(vol_size[0])
            y_flow = np.arange(vol_size[1])
            z_flow = np.arange(vol_size[2])
            grid = np.rollaxis(np.array((np.meshgrid(y_flow, x_flow, z_flow))), 0, 4)  # original coordinate
            grid_x = grid_sample(x, y, z, grid[:, :, :, 0], sample_num)
            grid_y = grid_sample(x, y, z, grid[:, :, :, 1], sample_num)
            grid_z = grid_sample(x, y, z, grid[:, :, :, 2], sample_num)  # X (10,10,10)

            sample = flow + grid
            sample_x = grid_sample(x, y, z, sample[:, :, :, 0], sample_num)
            sample_y = grid_sample(x, y, z, sample[:, :, :, 1], sample_num)
            sample_z = grid_sample(x, y, z, sample[:, :, :, 2], sample_num)  # Y (10,10,10)

            sum_x = np.sum(flow[:, :, :, 0])
            sum_y = np.sum(flow[:, :, :, 1])
            sum_z = np.sum(flow[:, :, :, 2])

            ave_x = sum_x / (vol_size[0] * vol_size[1] * vol_size[2])
            ave_y = sum_y / (vol_size[0] * vol_size[1] * vol_size[2])
            ave_z = sum_z / (vol_size[0] * vol_size[1] * vol_size[2])

            # formula
            Y = np.zeros((10, 10, 10, grid_dimension))
            X = np.zeros((10, 10, 10, grid_dimension))
            T = np.array([ave_x, ave_y, ave_z, 1])  # (4,1)
            # R = np.zeros((10, 10, 10, grid_dimension, grid_dimension))

            for i in np.arange(10):
                for j in np.arange(10):
                    for z in np.arange(10):
                        Y[i, j, z, :] = np.array([sample_x[i, j, z], sample_y[i, j, z], sample_z[i, j, z], 1])

            for i in np.arange(10):
                for j in np.arange(10):
                    for z in np.arange(10):
                        X[i, j, z, :] = np.array([grid_x[i, j, z], grid_y[i, j, z], grid_z[i, j, z], 1])

            X = X.reshape((1000, grid_dimension))
            Y = Y.reshape((1000, grid_dimension))
            R = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)  # R

            # build new grid(Use R to do the spatial transform)
            shifted_x = np.arange(vol_size[0])
            shifted_y = np.arange(vol_size[1])
            shifted_z = np.arange(vol_size[2])
            print(shifted_x.shape)
            print(shifted_y.shape)
            print(shifted_z.shape)
            shifted_grid = np.rollaxis(np.array((np.meshgrid(shifted_y, shifted_x, shifted_z))), 0, 4)
            print(shifted_grid.shape)
            for i in np.arange(vol_size[0]):
                for j in np.arange(vol_size[1]):
                    for z in np.arange(vol_size[2]):
                        coordinates = np.dot(R, np.array([i, j, z, 1]).reshape(4, 1)) + T.reshape(4, 1)
                        print("voxel." + '(' + str(i) + ',' + str(j) + ',' + str(z) + ')')
                        shifted_grid[i, j, z, 0] = coordinates[0]
                        shifted_grid[i, j, z, 1] = coordinates[1]
                        shifted_grid[i, j, z, 2] = coordinates[2]

            # interpolation
            xx = np.arange(vol_size[1])
            yy = np.arange(vol_size[0])
            zz = np.arange(vol_size[2])
            warp_seg = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], shifted_grid, method='nearest', bounds_error=False,
                               fill_value=0)

            # CVPR
            grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
            sample = flow + grid
            sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
            warp_seg2 = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False,
                                fill_value=0)

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
            warp_seg = nn_trf_model.predict([X_seg, pred])[0, ..., 0]

        # compute Volume Overlap (Dice)
        dice_vals[:, k] = dice(warp_seg, atlas_seg, labels=good_labels)
        print('%3d %5.3f %5.3f' % (k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k + 1]))))

        if save_file is not None:
            sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': good_labels})


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

if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test_miccai2018.py gpu_id model_dir iter_num
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])
