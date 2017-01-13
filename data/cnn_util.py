import caffe
import cv2
import numpy as np
import skimage

GPU_ID=1

def crop_image(x, target_height=227, target_width=227):
    print x
    # skimage.img_as_float convert image np.ndarray into float type, with range (0, 1)
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if image.ndim == 2:
        image = image[:,:,np.newaxis][:,:,[0,0,0]]  # convert the gray image to rgb image

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_width,target_height))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_width, target_height))


deploy = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-deploy.prototxt'
model = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-model.caffemodel'
mean = '/home/cl/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
# deploy = '/YOUR/OWN/PATH/TO/CAFFE/DEPLOY'
# model = '/YOUR/OWN/PATH/TO/CAFFEMODEL'
# mean = '/YOUR/OWN/CAFFE_ROOT_PATH/python/caffe/imagenet/ilsvrc_2012_mean.npy'

class CNN(object):

    def __init__(self, deploy=deploy, model=model, mean=mean, batch_size=100, width=227, height=227):

        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        return net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes)

        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch_file = image_list[start:end]
            image_batch = np.array(map(lambda x: crop_image(x, target_width=self.width, target_height=self.height), image_batch_file))

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]

            all_feats[start:end] = feats

            if end % 500 == 0:
                print "**********************************************************************"        
                print "***************** Already Preprocessed {0} images ********************".format(end)                      
                print "***********************************************************************"

        return all_feats

