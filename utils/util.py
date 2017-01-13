import numpy
from collections import OrderedDict
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import os
import warnings

# normalize 2-Dimension data
def normalize_feature(feat_origin):
    feat_shape = feat_origin.shape
    assert len(feat_shape) == 2, "this normalization for feature only suit 2-dimension feature"
    feat_norm = numpy.linalg.norm(feat_origin, axis=1)
    feat_norm[feat_norm < 1e-8] = 1
    feat_normalized = feat_origin / feat_norm[:, numpy.newaxis]
    return feat_normalized

def normalize_feature_theano(feat_origin):
    assert feat_origin.ndim == 2, "this normalization for feature only suit 2-dimension feature"
    feat_shape = feat_origin.shape
    feat_norm = feat_origin.norm(2, axis=1)
    feat_norm += 1e-8
    feat_normalized = feat_origin / feat_norm[:, None]
    return feat_normalized

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


'''
Theano uses shared variables for parameters, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk, borrow=True)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise ValueError('the parameter %s is not in the archive, so you should not use reload' % kk)
        params[kk] = pp[kk]

    return params

# load resnet-152/resnet-101 pretrain weight
def load_resnet_params(layer, name):
    assert name in ['res5b', 'res5c_branch2a', 'res5c_branch2b'], 'only suit for this feature maps'
    assert layer in ['resnet152', 'resnet101'], 'only suit for this two type net'
    if layer == 'resnet152':
        params = numpy.load("./cnn/resnet_152_params.npz")
    elif layer == 'resnet101':
        params = numpy.load("./cnn/resnet_101_params.npz")
    else:
        raise NotImplementedError

    ######################  5c_branch2a ####################
    res5c_branch2a_0 = theano.shared(params['res5c_branch2a_0'], name='res5c_branch2a_0', borrow=True)
    bn5c_branch2a_0 = theano.shared(params['bn5c_branch2a_0'], name='bn5c_branch2a_0', borrow=True)
    bn5c_branch2a_1 = theano.shared(params['bn5c_branch2a_1'], name='bn5c_branch2a_1', borrow=True)
    scale5c_branch2a_0 = theano.shared(params['scale5c_branch2a_0'], name='scale5c_branch2a_0', borrow=True)
    scale5c_branch2a_1 = theano.shared(params['scale5c_branch2a_1'], name='scale5c_branch2a_1', borrow=True)

    ######################  5c_branch2b ####################
    res5c_branch2b_0 = theano.shared(params['res5c_branch2b_0'], name='res5c_branch2b_0', borrow=True)
    bn5c_branch2b_0 = theano.shared(params['bn5c_branch2b_0'], name='bn5c_branch2b_0', borrow=True)
    bn5c_branch2b_1 = theano.shared(params['bn5c_branch2b_1'], name='bn5c_branch2b_1', borrow=True)
    scale5c_branch2b_0 = theano.shared(params['scale5c_branch2b_0'], name='scale5c_branch2b_0', borrow=True)
    scale5c_branch2b_1 = theano.shared(params['scale5c_branch2b_1'], name='scale5c_branch2b_1', borrow=True)

    #####################  5c_branch2c #####################
    res5c_branch2c_0 = theano.shared(params['res5c_branch2c_0'], name='res5c_branch2c_0', borrow=True)
    bn5c_branch2c_0 = theano.shared(params['bn5c_branch2c_0'], name='bn5c_branch2c_0', borrow=True)
    bn5c_branch2c_1 = theano.shared(params['bn5c_branch2c_1'], name='bn5c_branch2c_1', borrow=True)
    scale5c_branch2c_0 = theano.shared(params['scale5c_branch2c_0'], name='scale5c_branch2c_0', borrow=True)
    scale5c_branch2c_1 = theano.shared(params['scale5c_branch2c_1'], name='scale5c_branch2c_1', borrow=True) 

    conv_params = OrderedDict()
    bn_params = OrderedDict()

    if name in ['res5b']: 
        conv_params['res5c_branch2a_0'] = res5c_branch2a_0
        bn_params['bn5c_branch2a_0'] = bn5c_branch2a_0
        bn_params['bn5c_branch2a_1'] = bn5c_branch2a_1
        bn_params['scale5c_branch2a_0'] = scale5c_branch2a_0
        bn_params['scale5c_branch2a_1'] = scale5c_branch2a_1

    if name in ['res5b', 'res5c_branch2a']:
        conv_params['res5c_branch2b_0'] = res5c_branch2b_0
        bn_params['bn5c_branch2b_0'] = bn5c_branch2b_0
        bn_params['bn5c_branch2b_1'] = bn5c_branch2b_1
        bn_params['scale5c_branch2b_0'] = scale5c_branch2b_0
        bn_params['scale5c_branch2b_1'] = scale5c_branch2b_1

    if name in ['res5b', 'res5c_branch2a', 'res5c_branch2b']:
        conv_params['res5c_branch2c_0'] = res5c_branch2c_0
        bn_params['bn5c_branch2c_0'] = bn5c_branch2c_0
        bn_params['bn5c_branch2c_1'] = bn5c_branch2c_1
        bn_params['scale5c_branch2c_0'] = scale5c_branch2c_0
        bn_params['scale5c_branch2c_1'] = scale5c_branch2c_1
    
    return conv_params, bn_params

# some useful shorthands
def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# initialize the weights of parameters
def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def validate_options(options):
    # Put friendly reminders here
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    if options['use_dropout_lstm']:
        warnings.warn('dropout in the lstm seems not to help')

    # Other checks:
    if options['attn_type'] not in ["SCA"]:
        raise ValueError("the model is only for SCA model")

    if options['network'] not in ["resnet152"]:
        raise ValueError("this code is only suit for resnet model")

    return options

##########################  ResNet-152 Architecture #########################
def res5b_2_res5c_branch2a(res5b_res5b_relu_0_split_0, conv_params, bn_params):
    assert res5b_res5b_relu_0_split_0.ndim == 4
    res5c_branch2a = conv2d(input=res5b_res5b_relu_0_split_0, filters=conv_params['res5c_branch2a_0'], border_mode='valid', filter_flip=False)
    bn5c_branch2a = (res5c_branch2a - bn_params['bn5c_branch2a_0'].dimshuffle('x', 0, 'x', 'x')) / tensor.sqrt(bn_params['bn5c_branch2a_1'].dimshuffle('x', 0, 'x', 'x') + numpy.float32(1e-5))
    scale5c_branch2a = bn5c_branch2a * bn_params['scale5c_branch2a_0'].dimshuffle('x', 0, 'x', 'x') + bn_params['scale5c_branch2a_1'].dimshuffle('x', 0, 'x', 'x')
    res5c_branch2a_relu = tensor.nnet.relu(scale5c_branch2a, alpha=0.0)
    
    return res5c_branch2a_relu

def res5c_branch2a_2_res5c_branch2b(res5c_branch2a_relu, conv_params, bn_params):
    res5c_branch2b = conv2d(input=res5c_branch2a_relu, filters=conv_params['res5c_branch2b_0'], border_mode='half', filter_flip=False)
    bn5c_branch2b = (res5c_branch2b - bn_params['bn5c_branch2b_0'].dimshuffle('x', 0, 'x', 'x')) / tensor.sqrt(bn_params['bn5c_branch2b_1'].dimshuffle('x', 0, 'x', 'x') + numpy.float32(1e-5))
    scale5c_branch2b = bn5c_branch2b * bn_params['scale5c_branch2b_0'].dimshuffle('x', 0, 'x', 'x') + bn_params['scale5c_branch2b_1'].dimshuffle('x', 0, 'x', 'x')
    res5c_branch2b_relu = tensor.nnet.relu(scale5c_branch2b, alpha=0.0)

    return res5c_branch2b_relu

def res5c_branch2b_2_res5c_branch2c(res5c_branch2b_relu, conv_params, bn_params):
    res5c_branch2c = conv2d(input=res5c_branch2b_relu, filters=conv_params['res5c_branch2c_0'], border_mode='valid', filter_flip=False)
    bn5c_branch2c = (res5c_branch2c - bn_params['bn5c_branch2c_0'].dimshuffle('x', 0, 'x', 'x')) / tensor.sqrt(bn_params['bn5c_branch2c_1'].dimshuffle('x', 0, 'x', 'x') + numpy.float32(1e-5))
    scale5c_branch2c = bn5c_branch2c * bn_params['scale5c_branch2c_0'].dimshuffle('x', 0, 'x', 'x') + bn_params['scale5c_branch2c_1'].dimshuffle('x', 0, 'x', 'x')

    return scale5c_branch2c

def res5b_2_res5c(res5b, conv_params, bn_params):
    res5b_res5b_relu_0_split_0 = res5b
    res5b_res5b_relu_0_split_1 = res5b
    
    res5c_branch2a_relu = res5b_2_res5c_branch2a(res5b_res5b_relu_0_split_0, conv_params, bn_params)
    res5c_branch2b_relu = res5c_branch2a_2_res5c_branch2b(res5c_branch2a_relu, conv_params, bn_params)
    scale5c_branch2c = res5c_branch2b_2_res5c_branch2c(res5c_branch2b_relu, conv_params, bn_params)
    res5c = res5b_res5b_relu_0_split_1 + scale5c_branch2c
    res5c_relu = tensor.nnet.relu(res5c, alpha=0.0)

    return res5c_relu
