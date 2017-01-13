"""
__author__ = "chen long"

@article{chen2016sca,
  title={SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning},
  author={Chen, Long and Zhang, Hanwang and Xiao, Jun and Nie, Liqiang and Shao, Jian and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1611.05594},
  year={2016}
}

"""
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

import cPickle as pkl
import numpy
import copy
import os
import time

from collections import OrderedDict
from sklearn.cross_validation import KFold

import warnings

from utils.homogeneous_data import HomogeneousData

# supported optimizers
from utils.optimizers import adadelta, adam, rmsprop, sgd

# dataset iterators
from data import coco

import utils.metrics
from utils.util import *

theano.config.compute_test_value = 'off'

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'coco': (coco.load_data, coco.prepare_data)}


def get_dataset(name):
    return datasets[name][0], datasets[name][1]

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' %(pp, name)

# dropout in theano
def dropout_layer(state_before, use_noise, trng):
    """
    tensor switch is like an if statement that checks the
    value of the theano shared variable (use_noise), before
    either dropping out the state_before tensor or
    computing the appropriate activation. During training/testing
    use_noise is toggled on and off.
    """
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

"""
Neural network layer definitions.

The life-cycle of each of these layers is as follows
    1) The param_init of the layer is called, which creates
    the weights of the network.
    2) The fprop is called which builds that part of the Theano graph
    using the weights created in step 1). This automatically links
    these variables to the graph.

Each prefix is used like a key and should be unique
to avoid naming conflicts when building the graph.
"""
# layers: 'name': ('parameter initializer', 'fprop')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin is None:
        nin = options['dim']
    if nout is None:
        nout = options['dim']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    """
     Stack the weight matricies for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    # for the previous hidden activation
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    # if we are dealing with a mini-batch
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        init_state = tensor.alloc(0., n_samples, dim)
        init_memory = tensor.alloc(0., n_samples, dim)
    # during sampling
    else:
        n_samples = 1
        init_state = tensor.alloc(0., dim)
        init_memory = tensor.alloc(0., dim)

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # use the slice to calculate all the different gates
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[init_state, init_memory, None, None, None, None],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx_512=512, dimctx_2048=2048):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    # input to LSTM, similar to the above, we stack the matricies for compactness, do one
    # dot product, and use the slice function below to get the activations for each "gate"
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM to LSTM
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # context to LSTM, the input to the LSTM is the final 4096-dimension vector
    Wc = norm_weight(dimctx_2048,dim*4)
    params[_p(prefix,'Wc')] = Wc

    # attention: context -> hidden
    Wc_att_res5c_1 = norm_weight(1, 512)
    params[_p(prefix,'Wc_att_res5c_1')] = Wc_att_res5c_1

    Wc_att_res5c_2 = norm_weight(dimctx_2048, 512)
    params[_p(prefix,'Wc_att_res5c_2')] = Wc_att_res5c_2

    Wc_att_res5c_branch2b_1 = norm_weight(1, 512)
    params[_p(prefix, 'Wc_att_res5c_branch2b_1')] = Wc_att_res5c_branch2b_1

    Wc_att_res5c_branch2b_2 = norm_weight(dimctx_512, 512)
    params[_p(prefix, 'Wc_att_res5c_branch2b_2')] = Wc_att_res5c_branch2b_2

    # attention: LSTM -> hidden
    Wd_att_res5c_1 = norm_weight(dim, 512)
    params[_p(prefix,'Wd_att_res5c_1')] = Wd_att_res5c_1

    Wd_att_res5c_2 = norm_weight(dim, 512)
    params[_p(prefix,'Wd_att_res5c_2')] = Wd_att_res5c_2

    Wd_att_res5c_branch2b_1 = norm_weight(dim, 512)
    params[_p(prefix, 'Wd_att_res5c_branch2b_1')] = Wd_att_res5c_branch2b_1

    Wd_att_res5c_branch2b_2 = norm_weight(dim, 512)
    params[_p(prefix, 'Wd_att_res5c_branch2b_2')] = Wd_att_res5c_branch2b_2

    # attention: hidden bias
    b_att_res5c_1 = numpy.zeros((512,)).astype('float32')
    params[_p(prefix,'b_att_res5c_1')] = b_att_res5c_1

    b_att_res5c_2 = numpy.zeros((512,)).astype('float32')
    params[_p(prefix,'b_att_res5c_2')] = b_att_res5c_2

    b_att_res5c_branch2b_1 = numpy.zeros((512,)).astype('float32')
    params[_p(prefix,'b_att_res5c_branch2b_1')] = b_att_res5c_branch2b_1

    b_att_res5c_branch2b_2 = numpy.zeros((512,)).astype('float32')
    params[_p(prefix,'b_att_res5c_branch2b_2')] = b_att_res5c_branch2b_2

    # attention:
    U_att_res5c_1 = norm_weight(512,1)
    params[_p(prefix,'U_att_res5c_1')] = U_att_res5c_1
    c_att_res5c_1 = numpy.zeros((1,)).astype('float32')
    params[_p(prefix,'c_att_res5c_1')] = c_att_res5c_1

    U_att_res5c_2 = norm_weight(512,1)
    params[_p(prefix,'U_att_res5c_2')] = U_att_res5c_2
    c_att_res5c_2 = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att_res5c_2')] = c_att_res5c_2

    U_att_res5c_branch2b_1 = norm_weight(512, 1)
    params[_p(prefix,'U_att_res5c_branch2b_1')] = U_att_res5c_branch2b_1
    c_att_res5c_branch2b_1 = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att_res5c_branch2b_1')] = c_att_res5c_branch2b_1

    U_att_res5c_branch2b_2 = norm_weight(512, 1)
    params[_p(prefix,'U_att_res5c_branch2b_2')] = U_att_res5c_branch2b_2
    c_att_res5c_branch2b_2 = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att_res5c_branch2b_2')] = c_att_res5c_branch2b_2


    if options['selector']:
        # attention: selector
        W_sel = norm_weight(dim, 1)
        params[_p(prefix, 'W_sel')] = W_sel
        b_sel = numpy.float32(0.)
        params[_p(prefix, 'b_sel')] = b_sel

    return params

def lstm_cond_layer(tparams, conv_params, bn_params, state_below, options, prefix='lstm',
                    mask=None, context_res5c_branch2b=None, context_res5b=None, one_step=False,
                    init_memory=None, init_state=None,
                    trng=None, use_noise=None, sampling=True,
                    argmax=False, **kwargs):

    assert context_res5c_branch2b, 'Context must be provided'
    assert context_res5b, 'Context must be provided'

    branch2b_shape = context_res5c_branch2b.shape
    if context_res5c_branch2b.ndim == 4:
        context_res5c_branch2b_reshaped = context_res5c_branch2b.reshape([branch2b_shape[0], branch2b_shape[1], branch2b_shape[2]*branch2b_shape[3]]).transpose(0, 2, 1)
    elif context_res5c_branch2b.ndim ==3: 
        context_res5c_branch2b_reshaped = context_res5c_branch2b.reshape([branch2b_shape[0], branch2b_shape[1]*branch2b_shape[2]]).transpose(1, 0)
    else:
        raise ValueError("context dimension should be 3 or 4.")  

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # infer lstm dimension
    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected x
    # state_below is timesteps*num samples by d in training (TODO change to notation of paper)
    # this is n * d during sampling
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_, dp_=None, dp_att_=None):

        if context_res5c_branch2b_reshaped.ndim == 3:
            context_res5c_branch2b_reshaped_mean = context_res5c_branch2b_reshaped.mean(axis=1)
        elif context_res5c_branch2b_reshaped.ndim == 2:
            context_res5c_branch2b_reshaped_mean = context_res5c_branch2b_reshaped.mean(axis=0, keepdims=True)
        else:
            raise NotImplementedError       

        pctx_1_ = tensor.dot(context_res5c_branch2b_reshaped_mean[:, :, None], tparams[_p(prefix, 'Wc_att_res5c_branch2b_1')]) + tparams[_p(prefix, 'b_att_res5c_branch2b_1')]
        pstate_1_ = tensor.dot(h_, tparams[_p(prefix, 'Wd_att_res5c_branch2b_1')])
        pctx_1_ = pctx_1_ + pstate_1_[:, None, :]
        pctx_1_ = tanh(pctx_1_)

        alpha_1 = tensor.dot(pctx_1_, tparams[_p(prefix, 'U_att_res5c_branch2b_1')]) + tparams[_p(prefix, 'c_att_res5c_branch2b_1')]
        alpha_1_shp = alpha_1.shape
        alpha_1 = tensor.nnet.softmax(alpha_1.reshape([alpha_1_shp[0], alpha_1_shp[1]]))  # softmax 
     
        weighted_context_res5c_branch2b = context_res5c_branch2b_reshaped * alpha_1[:, None, :] * 512
        pctx_2_ = tensor.dot(weighted_context_res5c_branch2b, tparams[_p(prefix, 'Wc_att_res5c_branch2b_2')]) + tparams[_p(prefix, 'b_att_res5c_branch2b_2')]
        pstate_2_ = tensor.dot(h_, tparams[_p(prefix, 'Wd_att_res5c_branch2b_2')])
        pctx_2_ = pctx_2_ + pstate_2_[:, None, :]
        pctx_2_ = tanh(pctx_2_)
   
        alpha_2 = tensor.dot(pctx_2_, tparams[_p(prefix,'U_att_res5c_branch2b_2')])+tparams[_p(prefix, 'c_att_res5c_branch2b_2')]
        alpha_2_shp = alpha_2.shape
        alpha_2 = tensor.nnet.softmax(alpha_2.reshape([alpha_2_shp[0],alpha_2_shp[1]])) # softmax

        reshaped_alpha_2 = alpha_2.reshape([alpha_2_shp[0], 1, 7, 7])
        reshaped_alpha_1 = alpha_1.reshape([alpha_1_shp[0], alpha_1_shp[1], 1, 1])
        context_branch2b = context_res5c_branch2b * reshaped_alpha_2 * reshaped_alpha_1
        context_branch2b = context_branch2b * 49 * 512

        scale5c_branch2c = res5c_branch2b_2_res5c_branch2c(context_branch2b, conv_params, bn_params)
       
        context_res5c = scale5c_branch2c + context_res5b
        context_res5c = tensor.nnet.relu(context_res5c, alpha=0.0)

        res5c_shape = context_res5c.shape
        assert context_res5c.ndim == 4

        context_res5c_reshaped = context_res5c.reshape([res5c_shape[0], res5c_shape[1], res5c_shape[2]*res5c_shape[3]]).transpose(0, 2, 1)

        context_res5c_reshaped_mean = context_res5c_reshaped.mean(axis=1)
        pctx_3_ = tensor.dot(context_res5c_reshaped_mean[:, :, None], tparams[_p(prefix, 'Wc_att_res5c_1')]) + tparams[_p(prefix, 'b_att_res5c_1')]
        pstate_3_ = tensor.dot(h_, tparams[_p(prefix, 'Wd_att_res5c_1')])
        pctx_3_ = pctx_3_ + pstate_3_[:, None, :]
        pctx_3_ = tanh(pctx_3_)

        beta_1 = tensor.dot(pctx_3_, tparams[_p(prefix, 'U_att_res5c_1')]) + tparams[_p(prefix, 'c_att_res5c_1')]
        beta_1_shp = beta_1.shape
        beta_1 = tensor.nnet.softmax(beta_1.reshape([beta_1_shp[0], beta_1_shp[1]]))  # softmax

        weighted_context_res5c = context_res5c_reshaped * beta_1[:, None, :] * 2048
        pctx_4_ = tensor.dot(weighted_context_res5c, tparams[_p(prefix, 'Wc_att_res5c_2')]) + tparams[_p(prefix, 'b_att_res5c_2')]
        pstate_4_ = tensor.dot(h_, tparams[_p(prefix, 'Wd_att_res5c_2')])
        pctx_4_ = pctx_4_ + pstate_4_[:, None, :]
        pctx_4_ = tanh(pctx_4_)

        beta_2 = tensor.dot(pctx_4_, tparams[_p(prefix, 'U_att_res5c_2')] + tparams[_p(prefix, 'c_att_res5c_2')])
        beta_2_shp = beta_2.shape
        beta_2 = tensor.nnet.softmax(beta_2.reshape([beta_2_shp[0], beta_2_shp[1]]))

        reshaped_beta_2 = beta_2.reshape([beta_2_shp[0], 1, 7, 7])
        reshaped_beta_1 = beta_1.reshape([beta_1_shp[0], beta_1_shp[1], 1, 1])
        context_res5c = context_res5c * reshaped_beta_1 * reshaped_beta_2
        context_res5c = context_res5c * 49 * 2048

        ctx_ = pool.pool_2d(context_res5c, ds = (7, 7), ignore_border=True, mode='average_exc_pad')
        ctx_ = ctx_.reshape((ctx_.shape[0], ctx_.shape[1]))
        ctx_ = normalize_feature_theano(ctx_)        

        alpha_1_sample = alpha_1
        alpha_2_sample = alpha_2
        beta_1_sample = beta_1
        beta_2_sample = beta_2

        if options['selector']:
            sel_ = tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'W_sel')])+tparams[_p(prefix,'b_sel')])
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

        # Recover the activations to the lstm gates
        i = _slice(preact, 0, dim)
        f = _slice(preact, 1, dim)
        o = _slice(preact, 2, dim)
        if options['use_dropout_lstm']:
            i = i * _slice(dp_, 0, dim)
            f = f * _slice(dp_, 1, dim)
            o = o * _slice(dp_, 2, dim)
        i = tensor.nnet.sigmoid(i)
        f = tensor.nnet.sigmoid(f)
        o = tensor.nnet.sigmoid(o)
        c = tensor.tanh(_slice(preact, 3, dim))

        # compute the new memory/hidden state
        # if the mask is 0, just copy the previous state
        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        rval = [h, c, alpha_1, alpha_1_sample, alpha_2, alpha_2_sample, beta_1, beta_1_sample, beta_2, beta_2_sample, ctx_]
        if options['selector']:
            rval += [sel_]
        rval += [i, f, o, preact]
        return rval

    if options['use_dropout_lstm']:
        if options['selector']:
            _step0 = lambda m_, x_, dp_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_, sel_: \
                            _step(m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_, dp_)
        else:
            _step0 = lambda m_, x_, dp_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_: \
                            _step(m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_, dp_)
        dp_shape = state_below.shape
        if one_step:
            dp_mask = tensor.switch(use_noise,
                                    trng.binomial((dp_shape[0], 3*dim),
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], 3 * dim))
        else:
            dp_mask = tensor.switch(use_noise,
                                    trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
    else:
        if options['selector']:
            _step0 = lambda m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_, sel_: \
                            _step(m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_)
        else:
            _step0 = lambda m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_: \
                            _step(m_, x_, h_, c_, a_1, as_1, a_2, as_2, b_1, bs_1, b_2, bs_2, ct_)

    if one_step:
        if options['use_dropout_lstm']:
            if options['selector']:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, None, None, None, None, None, None, None)
            else:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, None, None, None, None, None, None)
        else:
            if options['selector']:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, None, None, None, None, None, None, None, None)
            else:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, None, None, None, None, None, None, None)
        return rval
    else:
        seqs = [mask, state_below]
        if options['use_dropout_lstm']:
            seqs += [dp_mask]
        outputs_info = [init_state,
                        init_memory,
                        tensor.alloc(0., n_samples, 512),
                        tensor.alloc(0., n_samples, 512),
                        tensor.alloc(0., n_samples, 49),
                        tensor.alloc(0., n_samples, 49),
                        tensor.alloc(0., n_samples, 2048),
                        tensor.alloc(0., n_samples, 2048),
                        tensor.alloc(0., n_samples, 49),
                        tensor.alloc(0., n_samples, 49),
                        tensor.alloc(0., n_samples, options['ctx_dim_2048'])]
        if options['selector']:
            outputs_info += [tensor.alloc(0., n_samples)]
        outputs_info += [None,
                         None,
                         None,
                         None,] 

        rval, updates = theano.scan(_step0,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps, profile=False)
        return rval, updates

# parameter initialization
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    ctx_dim_2048 = options['ctx_dim_2048']
    ctx_dim_512 = options['ctx_dim_512']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim_2048, nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim_2048, nout=options['dim'])
    # decoder: LSTM
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                       nin=options['dim_word'], dim=options['dim'],
                                       dimctx_512=ctx_dim_512, dimctx_2048=ctx_dim_2048)
    # potentially deep decoder (warning: should work but somewhat untested)
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            params = get_layer('ff')[0](options, params, prefix='ff_state_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
            params = get_layer('ff')[0](options, params, prefix='ff_memory_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
            params = get_layer('lstm_cond')[0](options, params, prefix='decoder_%d'%lidx,
                                               nin=options['dim'], dim=options['dim'],
                                               dimctx=ctx_dim)

    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
    if options['ctx2out']:
        params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim_2048, nout=options['dim_word'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'])
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, conv_params, bn_params, options, sampling=True):
    """ Builds the entire computational graph used for training
    """
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples,
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    ctx_res5b = tensor.tensor4('ctx_res5b', dtype='float32')
    ctx_res5c_branch2b = tensor.tensor4('ctx_res5c_branch2b', dtype='float32')

    x.tag.test_value = numpy.ones((10, 64), dtype='int64')
    mask.tag.test_value = numpy.random.random((10, 64)).astype('float32')
    ctx_res5b.tag.test_value = numpy.random.random((64, 2048, 7, 7)).astype('float32')
    ctx_res5c_branch2b.tag.test_value = numpy.random.random((64, 512, 7, 7)).astype('float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix, shift it forward in time
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    

    scale5c_branch2c = res5c_branch2b_2_res5c_branch2c(ctx_res5c_branch2b, conv_params, bn_params)
    res5c = ctx_res5b + scale5c_branch2c
    ctx_initial = tensor.nnet.relu(res5c, alpha=0.0)
    ctx_initial = pool.pool_2d(ctx_initial, ds = (7, 7), ignore_border=True, mode='average_exc_pad')
    ctx_initial = ctx_initial.reshape((ctx_initial.shape[0], ctx_initial.shape[1])) 

    ctx_initial = normalize_feature_theano(ctx_initial)

    init_state = get_layer('ff')[1](tparams, ctx_initial, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_initial, options, prefix='ff_memory', activ='tanh')
    # lstm decoder
    attn_updates = []
    proj, updates = get_layer('lstm_cond')[1](tparams, conv_params, bn_params, emb, options,
                                              prefix='decoder',
                                              mask=mask, 
                                              context_res5c_branch2b=ctx_res5c_branch2b,
                                              context_res5b=ctx_res5b,
                                              one_step=False,
                                              init_state=init_state,
                                              init_memory=init_memory,
                                              trng=trng,
                                              use_noise=use_noise,
                                              sampling=sampling)
    attn_updates += updates
    proj_h = proj[0]
    # optional deep attention
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh')
            init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh')
            proj, updates = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                                      prefix='decoder_%d'%lidx,
                                                      mask=mask, context=ctx0,
                                                      one_step=False,
                                                      init_state=init_state,
                                                      init_memory=init_memory,
                                                      trng=trng,
                                                      use_noise=use_noise,
                                                      sampling=sampling)
            attn_updates += updates
            proj_h = proj[0]

    # acturally parameter alpha_sample is not used
    alphas_1 = proj[2]
    alpha_1_sample = proj[3]

    alphas_2 = proj[4]
    alpha_2_sample = proj[5]

    betas_1 = proj[6]
    beta_1_sample = proj[7]

    betas_2 = proj[8]
    beta_2_sample = proj[9]    

    ctxs = proj[10]

    if options['selector']:
        sels = proj[11]

    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng)

    # compute word probabilities
    # the shape of proj_h is (#timestep, #samples, dim)
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)

    # compute softmax
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape  # logit_shp is (#ntimestep, #samples, n_word)
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # Index into the computed probability to give the log likelihood
    x_flat = x.flatten()
    p_flat = probs.flatten()
    # tensor.arange(x_flat.shape[0])*probs.shape[1] get all the first index of each row in p_flat, x_flat is the offset
    cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    masked_cost = cost * mask
    cost = (masked_cost).sum(0)

    # optional outputs
    opt_outs = dict()
    if options['selector']:
        opt_outs['selector'] = sels

    return trng, use_noise, [x, mask, ctx_res5b, ctx_res5c_branch2b], alphas_1, alpha_1_sample, alphas_2, alpha_2_sample, betas_1, beta_1_sample, betas_2, beta_2_sample, cost, opt_outs

# build a sampler
def build_sampler(tparams, conv_params, bn_params, options, use_noise, trng, sampling=True):
    """ Builds a sampler used for generating from the model
    Parameters
    ----------
        See build_model function above
    Returns
    -------
    f_init : theano function
        Input: annotation, Output: initial lstm state and memory
    f_next: theano function
        Takes the previous word/state/memory + ctx0 and runs ne
        step through the lstm (used for beam search)
    """
    # context: 2048 x 7 x 7
    ctx_res5b = tensor.tensor3('ctx_res5b', dtype='float32')
    # context: 512 x 7 x 7
    ctx_res5c_branch2b = tensor.tensor3('ctx_res5c_branch2b', dtype='float32')

    ctx_res5b.tag.test_value = numpy.random.random((2048, 7, 7)).astype('float32')
    ctx_res5c_branch2b.tag.test_value = numpy.random.random((512, 7, 7)).astype('float32')

    # initial state/cell
    ctx_initial = res5c_branch2b_2_res5c_branch2c(ctx_res5c_branch2b[None, :, :, :], conv_params, bn_params)
    ctx_initial = ctx_res5b[None, :, :, :] + ctx_initial
    ctx_initial = tensor.nnet.relu(ctx_initial, alpha=0.0)
    ctx_initial = pool.pool_2d(ctx_initial, ds = (7, 7), ignore_border=True, mode='average_exc_pad') 
    ctx_initial = normalize_feature_theano(ctx_initial[:, :, 0, 0])

    ctx_initial = ctx_initial.reshape((ctx_initial.shape[1],))    # resahpe ctx_initial into a vector
    assert ctx_initial.ndim == 1, "ctx_initial should reshape into a vector with dimension 1"

    init_state = [get_layer('ff')[1](tparams, ctx_initial, options, prefix='ff_state', activ='tanh')]
    init_memory = [get_layer('ff')[1](tparams, ctx_initial, options, prefix='ff_memory', activ='tanh')]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh'))
            init_memory.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh'))

    print 'Building f_init...',
    f_init = theano.function([ctx_res5b, ctx_res5c_branch2b], [ctx_res5b, ctx_res5c_branch2b]+init_state+init_memory, name='f_init', profile=False)
    print 'Done'

    # build f_next
    # ctx_res5b: 2048 x 7 x 7
    ctx_res5b = tensor.tensor3('ctx_res5b', dtype='float32')
    ctx_res5c_branch2b = tensor.tensor3('ctx_res5c_branch2b', dtype='float32')
    x = tensor.vector('x_sampler', dtype='int64')
    init_state = [tensor.matrix('init_state', dtype='float32')]
    init_memory = [tensor.matrix('init_memory', dtype='float32')]
 
    ctx_res5b.tag.test_value = numpy.random.random((2048, 7, 7)).astype('float32')
    ctx_res5c_branch2b.tag.test_value = numpy.random.random((512, 7, 7)).astype('float32')
    x.tag.test_value = numpy.ones((1, )).astype('int64')
    init_state[0].tag.test_value = numpy.random.random((1, 1000)).astype('float32')
    init_memory[0].tag.test_value = numpy.random.random((1, 1000)).astype('float32')

    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(tensor.matrix('init_state', dtype='float32'))
            init_memory.append(tensor.matrix('init_memory', dtype='float32'))

    # for the first word (which is coded with -1), emb should be all zero
    emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][x])

    proj = get_layer('lstm_cond')[1](tparams, conv_params, bn_params, emb, options,
                                     prefix='decoder',
                                     mask=None, 
                                     context_res5c_branch2b=ctx_res5c_branch2b,
                                     context_res5b=ctx_res5b,
                                     one_step=True,
                                     init_state=init_state[0],
                                     init_memory=init_memory[0],
                                     trng=trng,
                                     use_noise=use_noise,
                                     sampling=sampling)

    next_state, next_memory, ctxs = [proj[0]], [proj[1]], [proj[10]]
    proj_h = proj[0]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            proj = get_layer('lstm_cond')[1](tparams, conv_params, bn_params, proj_h, options,
                                             prefix='decoder_%d'%lidx,
                                             context=ctx,
                                             one_step=True,
                                             init_state=init_state[lidx],
                                             init_memory=init_memory[lidx],
                                             trng=trng,
                                             use_noise=use_noise,
                                             sampling=sampling)
            next_state.append(proj[0])
            next_memory.append(proj[1])
            ctxs.append(proj[10])
            proj_h = proj[0]

    if options['use_dropout']:
        proj_h = dropout_layer(proj[0], use_noise, trng)
    else:
        proj_h = proj[0]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs[-1], options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    f_next = theano.function([x, ctx_res5b, ctx_res5c_branch2b]+init_state+init_memory, [next_probs, next_sample]+next_state+next_memory, name='f_next', profile=False)

    return f_init, f_next

# generate sample
def gen_sample(tparams, f_init, f_next, ctx_res5b_0, ctx_res5c_branch2b_0, options,
               trng=None, k=1, maxlen=30, stochastic=False):
    """Generate captions with beam search.

    This function uses the beam search algorithm to conditionally
    generate candidate captions. Supports beamsearch and stochastic
    sampling.

    Parameters
    ----------
    tparams : OrderedDict()
        dictionary of theano shared variables represented weight
        matricies
    f_init : theano function
        input: annotation, output: initial lstm state and memory
    f_next: theano function
        takes the previous word/state/memory + ctx0 and runs one
        step through the lstm
    ctx0 : numpy array
        annotation from convnet
        [e.g (512 x 14 x14)]
    options : dict
        dictionary of flags and options
    trng : random number generator
    k : int
        size of beam search
    maxlen : int
        maximum allowed caption size
    stochastic : bool
        if True, sample stochastically

    Returns
    -------
    sample : list of list
        each sublist contains an (encoded) sample from the model
    sample_score : numpy array
        scores of each sample
    """
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []

    # only matters if we use lstm encoder
    rval = f_init(ctx_res5b_0.astype('float32'), ctx_res5c_branch2b_0.astype('float32'))
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[2+lidx])    ######### for the return value now are 2 #########
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[2+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx_res5b_0, ctx_res5c_branch2b_0]+next_state+next_memory))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        next_state = []
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

        if stochastic:
            sample.append(next_w[0]) # if we are using stochastic sampling this easy
            sample_score += next_p[0,next_w[0]]
            if next_w[0] == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll

            voc_size = next_p.shape[1]
            # indexing into the correct selected captions
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat] # extract costs from top hypothesis

            # a bunch of lists to hold future hypothesis
            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
            new_hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories.append([])

            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

            # check the finished samples for <eos> character
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])
            hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_memories.append([])

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1 # completed sample!
                else:
                    new_live_k += 1 # collect collect correct states/memories
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_states[lidx].append(new_hyp_states[lidx][idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(numpy.array(hyp_states[lidx]))
            next_memory = []
            for lidx in xrange(options['n_layers_lstm']):
                next_memory.append(numpy.array(hyp_memories[lidx]))

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def pred_probs(f_log_probs, options, worddict, prepare_data, data, iterator, verbose=False):
    """ Get log probabilities of captions
    Parameters
    ----------
    f_log_probs : theano function
        compute the log probability of a x given the context
    options : dict
        options dictionary
    worddict : dict
        maps words to one-hot encodings
    prepare_data : function
        see corresponding dataset class for details
    data : numpy array
        output of load_data, see corresponding dataset class
    iterator : KFold
        indices from scikit-learn KFold
    verbose : boolean
        if True print progress
    Returns
    -------
    probs : numpy array
        array of log probabilities indexed by example
    """
   
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')
    L = numpy.zeros((n_samples, 1))

    n_done = 0

    for _, valid_index in iterator:
        x, mask, ctx_res5b, ctx_res5c_branch2b = prepare_data([data[0][t] for t in valid_index],
                                                               data[1],
                                                               data[2],
                                                               worddict,
                                                               maxlen=None,
                                                               n_words=options['n_words'])
        pred_probs = f_log_probs(x,mask,ctx_res5b,ctx_res5c_branch2b)
        probs[valid_index] = pred_probs[:,None]

        L[valid_index] = mask.sum(0).reshape([mask.shape[1], 1])

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    perp = 2**( -1 *numpy.sum(probs) / numpy.sum(L) / numpy.log(2))
    
    return -1* numpy.mean(probs), perp

"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following:
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds
"""
def train(feat_name='res5c_branch2b',
          exp_name='test_1',
          finetune_conv_params=True,
          dim_word=100,  # word vector dimensionality
          ctx_dim_2048=2048,  
          ctx_dim_512=512,
          dim=1000,  # the number of LSTM units
          attn_type='SCA', 
          network = 'resnet152',   
          n_layers_out=1,  # number of layers used to compute logit
          n_layers_lstm=1,  # number of lstm layers
          prev2out=True,  # Feed previous word into logit
          ctx2out=False,  # Feed attention weighted ctx into logit
          patience=50,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.0001,  # weight decay coeff
          alpha_1_c=0.,  # doubly stochastic coeff for alpha
          alpha_2_c=0.,
          beta_1_c=0.,  # doubly stochastic coeff for beta
          beta_2_c=0.,
          lrate=0.01,  # used only for SGD
          selector=False,  # selector 
          maxlen=100,  # maximum length of the description
          optimizer='adadelta',
          batch_size = 64,
          valid_batch_size = 16,
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=500,  # generate some samples after every sampleFreq updates
          dataset='coco',
          use_dropout=True,  # setting this true turns on dropout at various points 
          use_dropout_lstm=False,  # dropout on lstm gates
          reload_=False,
          from_dir=None, # the directory of reload model, None for the origin directory
          save_per_epoch=False, # this saves down the model every epoch 
          save_dir='exp'): # the directory to save the model, default is in exp

    save_path = os.path.join(save_dir, exp_name)
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    # hyperparam dict
    model_options = locals().copy()
    model_options = validate_options(model_options)

    if reload_:
        if from_dir is None:
            from_dir = save_path
        else:
            from_dir = os.path.join(save_dir, from_dir)

    # reload options
#    if reload_:
#        options_saved = os.path.join(from_dir, "model.npz.pkl")
#        assert os.path.isfile(options_saved)
#        print "Reloading options"
#        with open(options_saved, 'rb') as f:
#            model_options = pkl.load(f)

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    train, valid, test, worddict = load_data()

    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    model_options['n_words'] = len(word_idict)   
  
    print "Using the following parameters:"
    print  model_options

    with open(os.path.join(save_path, 'model.npz.pkl'), "w") as f:
        pkl.dump(model_options, f)
    
    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    print 'Building model'
    params = init_params(model_options)
    if reload_:
        model_saved = os.path.join(from_dir, "model_best_so_far.npz")
        assert os.path.isfile(model_saved)
        print "Reloading model"
        params = load_params(model_saved, params)

    # numpy arrays -> theano shared variables
    tparams = init_tparams(params)    
    
    # load the resnet pretrain weights
    conv_params, bn_params = load_resnet_params(network, feat_name)
 
    if finetune_conv_params:
        if reload_:
            model_saved = os.path.join(from_dir, "model_best_so_far.npz")
            pp = numpy.load(model_saved)
            for kk, vv in conv_params.iteritems():
                if kk in pp:
                    conv_params[kk].set_value(pp[kk])

    # In order, we get:
    #   1) trng - theano random number generator
    #   2) use_noise - flag that turns on dropout
    #   3) inps - inputs for f_grad_shared
    #   4) cost - log likelihood for each sentence
    #   5) opts_out - optional outputs (e.g selector)
    trng, use_noise, \
          inps, \
          alphas_1, alphas_sample_1, \
          alphas_2, alphas_sample_2, \
          betas_1, beta_sample_1, \
          betas_2, beta_sample_2, \
          cost, \
          opt_outs = \
          build_model(tparams, conv_params, bn_params,  model_options)


    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 [see top right of page 4], 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation
    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, conv_params, bn_params, model_options, use_noise, trng)

    # we want the cost without any the regularizers
    f_log_probs = theano.function(inps, -cost, profile=False, updates=None)

    cost = cost.mean()
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # Doubly stochastic regularization
    # alphas shape is (#ntimestep, #samples, 196)
    if alpha_1_c > 0.:
        alpha_1_c = theano.shared(numpy.float32(alpha_1_c), name='alpha_1_c')
        alpha_1_reg = alpha_1_c * ((1.-alphas_1.sum(0))**2).mean(0).mean()
        cost += alpha_1_reg

    if alpha_2_c > 0.:
        alpha_2_c = theano.shared(numpy.float32(alpha_2_c), name='alpha_2_c')
        alpha_2_reg = alpha_2_c * ((1.-alphas_2.sum(0))**2).mean(0).mean()
        cost += alpha_2_reg

    if beta_1_c > 0.:
        beta_1_c = theano.shared(numpy.float32(beta_1_c), name='beta_1_c')
        beta_1_reg = beta_1_c * ((1.-betas_1.sum(0))**2).mean(0).mean()
        cost += beta_1_reg

    if beta_2_c > 0.:
        beta_2_c = theano.shared(numpy.float32(beta_2_c), name='beta_2_c')
        beta_2_reg = beta_2_c * ((1.-betas_2.sum(0))**2).mean(0).mean()
        cost += beta_2_reg

    hard_attn_updates = [] 
    # Backprop!
    if finetune_conv_params:
        tparams.update(conv_params)

    grads = tensor.grad(cost, wrt=itemlist(tparams))

    # to getthe cost after regularization or the gradients, use this
    # f_cost = theano.function([x, mask, ctx], cost, profile=False)
    # f_grad = theano.function([x, mask, ctx], grads, profile=False)

    # f_grad_shared computes the cost and updates adaptive learning rate variables
    # f_update updates the weights of the model
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, hard_attn_updates)

    print 'Optimization'

    # [See note in section 4.3 of paper]
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)

#    if train:
#        kf_train = KFold(len(train[0]), n_folds=len(train[0])/valid_batch_size, shuffle=False)
    if valid:
        kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=False)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    # history_errs is a bare-bones training log that holds the validation and test error
    history_errs = []
    history_score = []

    # reload history
#    if reload_:
#        print 'reload history errors and history score...'
#        history_errs = numpy.load(os.path.join(from_dir, 'model_best_so_far.npz'))['history_errs'].tolist()
#        history_score = numpy.load(os.path.join(from_dir, 'model_best_so_far.npz'))['history_score'].tolist()
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    uidx_best_valid_err = 0
    best_valid_err = -1
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for caps in train_iter:
            n_samples += len(caps)
            uidx += 1
            # turn on dropout
            use_noise.set_value(1.)

            # preprocess the caption, recording the
            # time spent to help detect bottlenecks
            pd_start = time.time()
            x, mask, ctx_res5b, ctx_res5c_branch2b = prepare_data(caps,
                                        train[1],
                                        train[2],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=model_options['n_words'])
            pd_duration = time.time() - pd_start

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            # get the cost for the minibatch, and update the weights
            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx_res5b, ctx_res5c_branch2b)
            f_update(lrate)
            ud_duration = time.time() - ud_start # some monitoring for each mini-batch

            # Numerical stability check
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if dispFreq > 0 and (uidx == 1 or numpy.mod(uidx, dispFreq) == 0):
                history_errs.append([eidx, uidx, cost, pd_duration, ud_duration])
                numpy.savetxt(os.path.join(save_path, "history_errs.txt"), history_errs, fmt="%.3f")
                print 'Epoch', eidx, 'Update', uidx, 'Cost', cost, 'PD', pd_duration, 'UD', ud_duration

            # Checkpoint
            if saveFreq > 0 and numpy.mod(uidx, saveFreq) == 0:
                pass

            # Print a generated sample as a sanity check
            if sampleFreq > 0 and numpy.mod(uidx, sampleFreq) == 0:
                # turn off dropout first
                use_noise.set_value(0.)
                x_s = x
                mask_s = mask
                ctx_res5b_s = ctx_res5b
                ctx_res5c_branch2b_s = ctx_res5c_branch2b
                # generate and decode the a subset of the current training batch
                for jj in xrange(numpy.minimum(10, len(caps))):
                    sample, score = gen_sample(tparams, f_init, f_next, 
                                               ctx_res5b_s[jj], ctx_res5c_branch2b_s[jj], model_options,
                                               trng=trng, k=5, maxlen=30, stochastic=False)
                    # Decode the sample from encoding back to words
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv],
                        else:
                            print 'UNK',
                    print

                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                print word_idict[vv],
                            else:
                                print 'UNK',
                    print

            # Log validation loss + checkpoint the model with the best validation log likelihood
            if validFreq > 0 and numpy.mod(uidx, validFreq) == 0:
                valid_start = time.time()
                valid_err = 0
                valid_perp = 0
  
                valid_err, valid_perp = pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid)
                if valid_perp < 40:
                    scores = utils.metrics.compute_score(valid, test, word_idict,
                                                   f_init, f_next, model_options,
                                                   beam=5, save_result_dir=save_path)

                    if scores['valid']:
                        valid_B1 = scores['valid']['Bleu_1']
                        valid_B2 = scores['valid']['Bleu_2']
                        valid_B3 = scores['valid']['Bleu_3']
                        valid_B4 = scores['valid']['Bleu_4']
                        valid_Rouge = scores['valid']['ROUGE_L']
                        valid_Cider = scores['valid']['CIDEr']
                        valid_meteor = scores['valid']['METEOR']
                    else:
                        valid_B1 = 0
                        valid_B2 = 0
                        valid_B3 = 0
                        valid_B4 = 0
                        valid_Rouge = 0
                        valid_Cider = 0
                        valid_meteor = 0

                    if scores['test']:
                        test_B1 = scores['test']['Bleu_1']
                        test_B2 = scores['test']['Bleu_2']
                        test_B3 = scores['test']['Bleu_3']
                        test_B4 = scores['test']['Bleu_4']
                        test_Rouge = scores['test']['ROUGE_L']
                        test_Cider = scores['test']['CIDEr']
                        test_meteor = scores['test']['METEOR']
                    else:
                        test_B1 = 0
                        test_B2 = 0
                        test_B3 = 0
                        test_B4 = 0
                        test_Rouge = 0
                        test_Cider = 0
                        test_meteor = 0
 
                    history_score.append([eidx, uidx,
                                         valid_B1, valid_B2, valid_B3, valid_B4,
                                         valid_Rouge, valid_Cider, valid_meteor,
                                         test_B1, test_B2, test_B3, test_B4,
                                         test_Rouge, test_Cider, test_meteor,
                                         valid_err, valid_perp])

                    numpy.savetxt(os.path.join(save_path, "history_score.txt"), history_score, fmt="%.3f")

                    if len(numpy.array(history_score)) > 1 and (valid_B4 > numpy.array(history_score)[:-1, 5].max() or valid_meteor > numpy.array(history_score)[:-1, 8].max() or valid_Cider > numpy.array(history_score)[:-1, 7].max()):
                        print 'Saving model with best test Bleu_4/Meteor/Cider score'
                        current_params = unzip(tparams)
                        params_best_bleu = copy.copy(current_params)
                        numpy.savez(os.path.join(save_path, "model_best_bleu.npz"), history_errs=history_errs, history_score=history_score, **params_best_bleu)

                    if len(numpy.array(history_score)) > 1 and valid_err < numpy.array(history_score)[:-1, 16].min():
                        best_p = unzip(tparams)
                        print 'Saving best model so far'
                        params = copy.copy(best_p)
                        numpy.savez(os.path.join(save_path, "model_best_so_far.npz"), history_errs=history_errs, history_score=history_score, **params)
                        bad_counter = 0
                        best_valid_err = valid_err
                        uidx_best_valid_err = uidx
                    elif len(numpy.array(history_score)) > 1 and valid_err >= numpy.array(history_score)[:-1, 16].min():
                        bad_counter += 1
                        print 'history best ', numpy.array(history_score)[:, 16].min()
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    print("bad_counter: %d" % bad_counter)

                valid_duration = time.time() - valid_start
                print("validation use time: %f" % valid_duration)
                print("validation loss: %f, validation perp: %f" % (valid_err, valid_perp))

        print 'Seen %d samples' % n_samples

        if estop:
            break

        if save_per_epoch:
            numpy.savez(saveto + '_epoch_' + str(eidx + 1), history_errs=history_errs, history_score=history_score, **unzip(tparams))

    # use the best nll parameters for final checkpoint (if they exist)
    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = 0
    test_err = 0
    valid_perp = 0
    test_perp = 0

    if valid:
        valid_err, valid_perp = pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid)
    if test:
        test_err, test_perp  = pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test)

    print   'valid_err', valid_err, 'valid_perp', valid_perp, 'test_err', test_err, 'test_perp', test_perp

if __name__ == '__main__':
    train()
