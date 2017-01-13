import cPickle as pkl
import os
import sys
import time

from scipy import sparse
import numpy
import h5py

def prepare_data(caps, res5b_features, res5c_branch2b_features, worddict, maxlen=None, n_words=10000, zero_pad=False):
    # x: a list of sentences
    seqs = []
    res5b_feat_list = []
    res5c_branch2b_feat_list = []
    for cc in caps:  # cc is a tuple with 2 elements, the first is a cpation, the second is the image number
        try:
            seqs.append([worddict[w.lower()] if (w.lower() in worddict and worddict[w.lower()] < n_words) else 1 for w in cc[0].split()])
            res5b_feat_list.append(res5b_features[cc[1]])
            res5c_branch2b_feat_list.append(res5c_branch2b_features[cc[1]])
        except:
            raise ValueError("don't know why")

    lengths = [len(s) for s in seqs]

    if maxlen != None: # and numpy.max(lengths) >= maxlen:
        new_seqs = []
        new_res5b_feat_list = []
        new_res5c_branch2b_feat_list = []
        new_lengths = []
        for l, s, y, z in zip(lengths, seqs, res5b_feat_list, res5c_branch2b_feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_res5b_feat_list.append(y)
                new_res5c_branch2b_feat_list.append(z)
                new_lengths.append(l)
        lengths = new_lengths
        res5b_feat_list = new_res5b_feat_list
        res5c_branch2b_feat_list = new_res5c_branch2b_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None, None

    y_res5b = numpy.array(res5b_feat_list)
    y_res5c_branch2b = numpy.array(res5c_branch2b_feat_list)

    assert not zero_pad,"code below is not suit for 4-dimension y"
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y_res5b, y_res5c_branch2b

def load_data(load_train=True, load_dev=True, load_test=True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    res5b_path = './data/coco/layer152/res5b/'
    res5c_branch2b_path = './data/coco/layer152/res5c_branch2b/'

    save_path = os.getcwd()
    if not os.path.exists(os.path.join(save_path , res5b_path)):
      os.chdir(os.path.dirname(save_path))  # so the path ./ can show the current directory
    #############
    # LOAD DATA #
    #############
    print '... loading data'

    start_time = time.time()

    res5b_h5_f = h5py.File(res5b_path + 'image_data.h5','r')
    res5c_branch2b_h5_f = h5py.File(res5c_branch2b_path + 'image_data.h5', 'r')

    def load_cap(dataset):
        with open(res5b_path + 'coco_align.'+ dataset +'.pkl', 'rb') as f:
            dataset_cap = pkl.load(f)
        res5b_img = res5b_h5_f.get(dataset)
        res5c_branch2b_img = res5c_branch2b_h5_f.get(dataset)
        return dataset_cap, res5b_img, res5c_branch2b_img

    datasets = ['train', 'dev', 'test']

    rval = []    
    for dataset in datasets:
        if eval('load_'+dataset):
            cap = load_cap(dataset)
            print "already load " + dataset +" set"
        else:
            print "don't load " + dataset +" set"
            cap = None
        rval.append(cap)

    with open(res5b_path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)
        rval.append(worddict)
    print '... load worddict'

    os.chdir(save_path)

    end_time = time.time()
    print "load the coco dataset has spend time: %f" %(end_time - start_time) 

    return rval

if __name__ == '__main__':
    train, valid, test, worddict = load_data()
    print("train set size is: %i" %len(train[0]))
    print train[1]
    print train[2]
    print("valid set size is: %i" %len(valid[0]))
    print valid[1]
    print valid[2]
    print("test set size is: %i" %len(test[0]))
    print test[1]
    print test[2]
    print("word dictionary size is %i" %len(worddict))
