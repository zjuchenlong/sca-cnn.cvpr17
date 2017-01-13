from cnn_util import *
import numpy as np
import os
import scipy
import json
import cPickle
import string
import h5py

res_deploy_path = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-deploy.prototxt'
res_model_path = '/home/cl/dataset/caffemodel/ResNet/ResNet-152-model.caffemodel'
coco_json_path = '/home/cl/dataset/COCO/dataset_coco.json'
image_path = '/home/cl/dataset/COCO'
# res_deploy_path = '/YOUR/OWN/PATH/TO/CAFFE/DEPLOY'
# res_model_path = '/YOUR/OWN/PATH/TO/CAFFEMODEL'
# coco_json_path = '/YOUR/OWN/PATH/TO/COCO_JSON'
# image_path = 'YOUR/OWN/PATH/TO/COCO/DATASET'

data = json.load(open(coco_json_path, "r"))

word_count = {}

train_idx = 0
test_idx = 0
dev_idx = 0

cap_train = []
cap_dev = []
cap_test = []

images_train = []
images_dev = []
images_test = []

for image in data['images']:

    split = image['split']
    filepath = image['filepath']
    filename = os.path.join(image_path, filepath, image['filename'])
    for sent in image['sentences']:
        sent_token = " ".join(sent['tokens'])
        if split == 'train':
            cap_train.append((sent_token, train_idx))

            # vocabulary is generated from train set
            for word in sent['tokens']:
                word_count[word] = word_count.get(word, 0) + 1

        elif split == 'test':
            cap_test.append((sent_token, test_idx))
        elif split == 'val':
            cap_dev.append((sent_token, dev_idx))
        elif split == 'restval':
            pass
        else:
            raise ValueError("don't know the split")
    
    if split == 'train':
        train_idx += 1
        images_train.append(filename)
    elif split == 'test':
        test_idx += 1
        images_test.append(filename)
    elif split == 'val':
        dev_idx += 1
        images_dev.append(filename)
    elif split == 'restval':
        pass
    else:
        raise ValueError("don't know the split")

# filter the word when apperance is less than 5
dictionary = {}
index = 2   # the dictionary value start from 2
dict_threshold = 5
for key, value in word_count.iteritems():
    if value < dict_threshold:
        pass
    else:
        dictionary[key] =index
        index = index + 1

feature_layers = ['res5c_branch2b', 'res5b'] 

for feature_layer in feature_layers:
    dir_path = './coco/layer152/' + feature_layer
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        continue

    with open(dir_path + '/dictionary.pkl', 'wb') as f:
        cPickle.dump(dictionary, f)

    datasets = ['train', 'test', 'dev']
    for dataset in datasets:
        with open(dir_path + '/coco_align.' + dataset + '.pkl', 'wb') as f:
            cPickle.dump(eval('cap_'+dataset), f)

    cnn = CNN(deploy = res_deploy_path,
              model = res_model_path,
              batch_size = 20,
              width = 224,
              height = 224)

    f = h5py.File(dir_path + '/image_data.h5', "w")

    for dataset in datasets:
        images_set = eval('images_'+dataset)

        if feature_layer in ['res5c_branch2a', 'res5c_branch2b']:
            image_feat = cnn.get_features(images_set, layers=feature_layer, layer_sizes=[512, 7, 7])
        elif feature_layer in ['res5b', 'res5c']:
            image_feat = cnn.get_features(images_set, layers=feature_layer, layer_sizes=[2048, 7, 7])
        elif feature_layer in ['pool5']:
            image_feat = cnn.get_features(images_set, layers=feature_layer, layer_sizes=[2048, 1, 1])
            image_feat = image_feat.reshape(image_feat.shape[0], imaeg_feat.shape[1])
        else:
            raise NotImplementedError('This feature layer is not implemented')

        f.create_dataset(dataset, shape = image_feat.shape, data = image_feat, dtype='float32')
        print("Finish process dataset "+dataset)

    f.close()

