# third party
import numpy as np
from keras import backend as K
from keras.legacy import interfaces
import keras
from keras.layers import Layer
import tensorflow as tf

# local
from .utils import transform, integrate_vec, affine_to_shift


class VecInt(Layer):

    def __init__(self, indexing='ij', method='ode', int_steps=7, **kwargs):
        """
        Vector Integration Layer
        
        Parameters:
            method can be any of the methods in neuron.utils.integrate_vec
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        # confirm built
        self.built = True

    def call(self, inputs):
        
        # prepare location shift
        loc_shift = inputs
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        # map transform across batch
        return tf.map_fn(self._single_int, loc_shift, dtype=tf.float32)

    def _single_int(self, inputs):

        vel = inputs
        return integrate_vec(vel, method=self.method,
                      nb_steps=self.int_steps,
                      ode_args={'rtol':1e-6, 'atol':1e-12},
                      time_pt=1)


class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms. 
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from 
    the identity matrix.


    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self, interp_method='linear', indexing='ij', **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N+1 x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1]. 
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([f == (self.ndims+1) for f in trf_shape]))

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d' 
                                % (self.ndims, trf_shape[-1]))
            # if not trf_shape[:-1] == vol_shape:
                # raise Exception('flow field size expected' + str(vol_shape) + \
                                # 'Found:' + str(trf_shape[:-1]))

        # confirm built
        self.built = True


    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # if not self.is_affine:
        #     if not all([trf.shape[1:-1][f] == vol.shape[1:-1][f] for f in range(self.ndims)]):
        #         raise Exception('Shift shape should match vol shape. '
        #                         'Got: ' + str(trf.shape[1:-1]) + ' and ' + str(vol.shape[1:-1]))

        # go from affine
        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        return tf.map_fn(self._single_transform, [inputs[0], trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

            # the code below is unnecessary.
            # zero_one = tf.concat([tf.zeros((1, self.ndims)), tf.ones((1,1))], axis=1)
            # trf = tf.concat([trf, zero_one], axis=0)
        # note this is unnecessarily extra graph since at every batch_entry location we have a tf.eye graph
        trf += tf.eye(self.ndims+1)[:self.ndims,:]  # add identity, hence affine is a shift from identitiy
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method)


class LocalBiasLayer(Layer):
    """ 
    local bias layer
    
    A layer with an additive bias at each volume element
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBiasLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinearLayer(Layer):
    """ 
    local linear layer
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalLinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalLinearLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias 

    def compute_output_shape(self, input_shape):
        return input_shape
 

class LocallyConnected3D(Layer):
    """
    code based on LocallyConnected3D from keras layers:
    https://github.com/keras-team/keras/blob/master/keras/layers/local.py

    # TODO: Comment better. Right now we have the comments from the 3D version from keras.

    Locally-connected layer for 3D inputs.
    The `LocallyConnected3D` layer works similarly
    to the `Conv3D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.
    # Examples
    ```python
        # apply a 3x3x3 unshared weights convolution with 64 output filters on a 32x32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected3D(64, (3, 3, 3), input_shape=(32, 32, 32, 1)))
        # now model.output_shape == (None, 30, 30, 30, 64)
        # notice that this layer will consume (30*30*30)*(3*3*3*1*64) + (30*30*30)*64 parameters
        # add a 3x3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected3D(32, (3, 3, 3)))
        # now model.output_shape == (None, 28, 28, 28, 32)
    ```
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    @interfaces.legacy_conv3d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
        super(LocallyConnected3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 3, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected3D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=5)

    def build(self, input_shape):
        
        if self.data_format == 'channels_last':
            input_row, input_col, input_z = input_shape[1:-1]
            input_filter = input_shape[4]
        else:
            input_row, input_col, input_z = input_shape[2:]
            input_filter = input_shape[1]
        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected3D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        output_z = conv_utils.conv_output_length(input_z, self.kernel_size[2],
                                                   self.padding, self.strides[2])
        self.output_row = output_row
        self.output_col = output_col
        self.output_z = output_z
        self.kernel_shape = (output_row * output_col * output_z,
                             self.kernel_size[0] *
                             self.kernel_size[1] *
                             self.kernel_size[2] * input_filter,
                             self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(output_row, output_col, output_z, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=5, axes={1: input_filter})
        else:
            self.input_spec = InputSpec(ndim=5, axes={-1: input_filter})
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            z = input_shape[4]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            z = input_shape[3]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])
        z = conv_utils.conv_output_length(z, self.kernel_size[2],
                                             self.padding, self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], self.filters, rows, cols, z)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, z, self.filters)

    def call(self, inputs):
        
        output = self.local_conv3d(inputs,
                                self.kernel,
                                self.kernel_size,
                                self.strides,
                                (self.output_row, self.output_col, self.output_z),
                                self.data_format)

        if self.use_bias:
            output = K.bias_add(output, self.bias,
                                data_format=self.data_format)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(
            LocallyConnected3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def local_conv3d(self, inputs, kernel, kernel_size, strides, output_shape, data_format=None):
        """Apply 3D conv with un-shared weights.
        # Arguments
            inputs: 4D tensor with shape:
                    (batch_size, filters, new_rows, new_cols)
                    if data_format='channels_first'
                    or 4D tensor with shape:
                    (batch_size, new_rows, new_cols, filters)
                    if data_format='channels_last'.
            kernel: the unshared weight for convolution,
                    with shape (output_items, feature_dim, filters)
            kernel_size: a tuple of 2 integers, specifying the
                        width and height of the 3D convolution window.
            strides: a tuple of 2 integers, specifying the strides
                    of the convolution along the width and height.
            output_shape: a tuple with (output_row, output_col)
            data_format: the data format, channels_first or channels_last
        # Returns
            A 4d tensor with shape:
            (batch_size, filters, new_rows, new_cols)
            if data_format='channels_first'
            or 4D tensor with shape:
            (batch_size, new_rows, new_cols, filters)
            if data_format='channels_last'.
        # Raises
            ValueError: if `data_format` is neither
                        `channels_last` or `channels_first`.
        """
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        stride_row, stride_col, stride_z = strides
        output_row, output_col, output_z = output_shape
        kernel_shape = K.int_shape(kernel)
        _, feature_dim, filters = kernel_shape

        xs = []
        for i in range(output_row):
            for j in range(output_col):
                for k in range(output_z):
                    slice_row = slice(i * stride_row,
                                    i * stride_row + kernel_size[0])
                    slice_col = slice(j * stride_col,
                                    j * stride_col + kernel_size[1])
                    slice_z = slice(k * stride_z,
                                    k * stride_z + kernel_size[2])
                    if data_format == 'channels_first':
                        xs.append(K.reshape(inputs[:, :, slice_row, slice_col, slice_z],
                                        (1, -1, feature_dim)))
                    else:
                        xs.append(K.reshape(inputs[:, slice_row, slice_col, slice_z, :],
                                        (1, -1, feature_dim)))

        x_aggregate = K.concatenate(xs, axis=0)
        output = K.batch_dot(x_aggregate, kernel)
        output = K.reshape(output,
                        (output_row, output_col, output_z, -1, filters))

        if data_format == 'channels_first':
            output = K.permute_dimensions(output, (3, 4, 0, 1, 2))
        else:
            output = K.permute_dimensions(output, (3, 0, 1, 2, 4))
        return output

