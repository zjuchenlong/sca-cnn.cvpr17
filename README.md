# SCA-CNN

> Source code for the paper: [SCA-CNN: Spatial and Channel-wise Attention in Convolution Networks for Imgae Captioning](https://arxiv.org/abs/1611.05594)

This code is based on [arctic-captions](https://github.com/kelvinxu/arctic-captions) and [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid).

This code is only for two-layered attention model in ResNet-152 Network for MS COCO dataset. Other networks (VGG-19) or datasets (Flickr30k/Flickr8k) can also be used with minor modifications.

## Dependencies
* A python library: [Theano](http://www.deeplearning.net/software/theano/).

* Other python package dependencies like **numpy/scipy, skimage, opencv, sklearn, hdf5** which can be installed by `pip`, or simply run  
  ~~~
  $ pip install -r requirements.txt
  ~~~

* [Caffe](http://caffe.berkeleyvision.org/) for image CNN feature extraction. You should install caffe and building the pycaffe interface to extract the image CNN feature. 

* The official coco evaluation scrpits [coco-caption](https://github.com/tylin/coco-caption) for results evaluation. Install it by simply adding it into `$PYTHONPATH`.

## Getting Started
1. **Get the code** `$ git clone` the repo and install the dependencies

2. **Save the pretrained CNN weights** Save the ResNet-152 weights pretrained on ImageNet. Before running the code, set the variable *deploy* and *model* in *save_resnet_weight.py* to your own path. Then run:
  ~~~
  $ cd cnn
  $ python save_resnet_weight.py
  ~~~
3. **Preprocessing the dataset** For the preprocessing of captioning, we directly use the processed JSON blob from [neuraltalk](http://cs.stanford.edu/people/karpathy/deepimagesent/). Similar to step 2, set the `PATH` in *cnn_until.py* and *make_coco.py* to your own install path. Then run:
  ~~~
  $ cd data
  $ python make_coco.py
  ~~~
4. **Training**  The results are saved in the directory `exp`.
  ~~~
  $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sca_resnet_branch2b.py
  ~~~

## Citation

If you find this code useful, please cite the following paper:

  ```
  @inproceedings{chen2016sca,
    title={SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning},
    author={Chen, Long and Zhang, Hanwang and Xiao, Jun and Nie, Liqiang and Shao, Jian and Liu, Wei and Chua, Tat-Seng},
    booktitle={CVPR},
    year={2017}
  }
  ```
