"""
single atlas segmentation based on Voxelmorph and Neuron
train the model to warp the atlas onto the volume, then the volume is labeled
"""
'''
# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model


import datagenerators
import networks
import losses


vol_size = (160, 192, 224)
base_data_dir = '/home/ys895/resize256/resize256-crop_x32/'
#find all the path of .npz file in the directory
#read training data
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
#shuffle the path of .npz file
#shuffle the training data
random.shuffle(train_vol_names)

#read atlas data
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol']
#add two more dimension into the atlas data
atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))

def train(model, gpu_id, lr, n_iterations, reg_param, model_save_iter, load_iter):

    model_dir = '/home/ys895/SAS_Models'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))


    # UNET filters
    nf_enc = [16,32,32,32]
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,3]
    else:
        nf_dec = [32,32,32,32,32,16,16,3]

    with tf.device(gpu):
        model = networks.unet(vol_size, nf_enc, nf_dec)
        if(load_iter != 0):
            net.load_weights('/home/ys895/SAS_Models/' + str(load_iter) + '.h5')

        model.compile(optimizer=Adam(lr=lr), loss=[
                      losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    # return the data, add one more dimension into the data
    train_example_gen = datagenerators.example_gen(train_vol_names)
    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))


    # In this part, the code inputs the data into the model
    # Before this part, the model was set
    for step in range(1, n_iterations+1):

       #Parameters for training : X(train_vol) ,atlas_vol(atlas) ,zero_flow
        X = train_example_gen.__next__()[0]
        train_loss = model.train_on_batch(
            [atlas_vol, X], [X, zero_flow])

        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        printLoss(step, 1, train_loss)

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(load_iter+step) + '.h5')


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model",
                        choices=['vm1','vm2'],default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4,help="learning rate")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=15000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=500,
                        help="frequency of model saves")
    parser.add_argument("--load_iter", type=int,
                        dest="load_iter", default=0,
                        help="the iteratons of models to load")

    args = parser.parse_args()
    train(**vars(args))

'''



"""
multi atlas segmentation based on Voxelmorph and Neuron

"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model


import datagenerators
import networks
import losses


vol_size = (160, 192, 224)
# train data preparation
base_data_dir = '/home/ys895/resize256/resize256-crop_x32/'
# find all the path of .npz file in the directory
# read training data
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
# shuffle the path of .npz file
# shuffle the training data
random.shuffle(train_vol_names)

# read the only one atlas data
#atlas = np.load('../data/atlas_norm.npz')
#atlas_vol = atlas['vol']

# add two more dimension into the atlas data
#atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))

# atlas_list: several atlas were read
atlas_file = open('../data/MAS_atlas.txt')
atlas_strings = atlas_file.readlines()
lenn = 1
atlas_list = list()
for i in range(0,lenn):
    st = atlas_strings[i]
    atlas_add = np.load(st.strip())
    atlas_add = atlas_add['vol_data']
    atlas_add = np.reshape(atlas_add,(1,)+atlas_add.shape+(1,))
    atlas_list.append(atlas_add)

# read atlas_norm as atlas used for training
#atlas = np.load('../data/atlas_norm.npz')
#atlas = atlas['vol']
#atlas = np.reshape(atlas,(1,)+atlas.shape+(1,))
#atlas_list.append(atlas)

list_num = len(atlas_list)

def train(model, gpu_id, lr, n_iterations, reg_param, model_save_iter, load_iter):

    model_dir = '/home/ys895/SAS_Models'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))


    # UNET filters
    nf_enc = [16,32,32,32]
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,3]
    else:
        nf_dec = [32,32,32,32,32,16,16,3]

    with tf.device(gpu):
        model = networks.unet(vol_size, nf_enc, nf_dec)
        if(load_iter != 0):
            model.load_weights('/home/ys895/SAS_Models/' + str(load_iter) + '.h5')

        model.compile(optimizer=Adam(lr=lr), loss=[
                      losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    # return the data, add one more dimension into the data
    train_example_gen = datagenerators.example_gen(train_vol_names)
    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))


    # In this part, the code inputs the data into the model
    # Before this part, the model was set
    for step in range(1, n_iterations+1):
       # choose randomly one of the atlas from the atlas_list
        rand_num = random.randint(0, list_num-1)
        atlas_vol = atlas_list[rand_num]

       #Parameters for training : X(train_vol) ,atlas_vol(atlas) ,zero_flow
        X = train_example_gen.__next__()[0]
        train_loss = model.train_on_batch(
            [atlas_vol, X], [X, zero_flow])

        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        printLoss(step, 1, train_loss)

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(load_iter+step) + '.h5')


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model",
                        choices=['vm1','vm2'],default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4,help="learning rate")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=15000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=500,
                        help="frequency of model saves")
    parser.add_argument("--load_iter", type=int,
                        dest="load_iter", default=0,
                        help="the iteratons of models to load")

    args = parser.parse_args()
    train(**vars(args))


