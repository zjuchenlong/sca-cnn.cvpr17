import caffe
import numpy as np
from collections import OrderedDict
import os

# deploy = '/YOUR/OWN/PATH/TO/RESNET-152/DEPLOY/'
# model = '/YOUR/OWN/PATH/TO/RESNET-152/CAFFEMODEL'
deploy = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-deploy.prototxt'
model = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-model.caffemodel'

net = caffe.Net(deploy, model, caffe.TEST)

def _p(a, b):
    return ("%s_%s" %(a, b))

params_ = OrderedDict()
for key, value in net.params.iteritems():
    for i in xrange(len(value)):
        params_[_p(key, i)] = value[i].data

np.savez("resnet_152_params", **params_)

print "resnet parameters had been saved"

