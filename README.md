# SCA-CNN

> Source code for the paper: [SCA-CNN: Spatial and Channel-wise Attention in Convolution Networks for Imgae Captioning](https://arxiv.org/abs/1611.05594)

Joint collaboration between the Zhejiang University & National University of Singapore &  Shandong University

This code is modified based on two previous works [arctic-captions](https://github.com/kelvinxu/arctic-captions) and [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid).

This code is only for two layers attention model in ResNet-152 Network for MS COCO dataset. Other networks (VGG-19) or other datasets (Flickr30k/Flickr8k) can also be used by some little modifications.

## Dependencies
1. This code is written in python with powerful [Theano](http://www.deeplearning.net/software/theano/) library.

2. Some other python package dependencies like **numpy/scipy, skimage, opencv, sklearn, hdf5** module can be installed by `pip`, or directly run command `$ pip install -r requirements.txt`

3. For image CNN feature extraction, we use [Caffe](http://caffe.berkeleyvision.org/). You should install caffe and building the pycaffe interface to extract the image CNN feature. 

4. For results evaluation, we use the official coco evaluation scrpits [coco-caption](https://github.com/tylin/coco-caption). Install it by simply adding it into `$PYTHONPATH`.

## Getting Started
- **Get the code** `$ git clone` the repo and install the dependencies

- **Save the pretrained CNN weights** Save the ResNet-152 weights pretrained on ImageNet. Before running the code, set the variable *deploy* and *model* in *save_resnet_weight.py* to your own path. Then run:
```bash
$ cd cnn
$ python save_resnet_weight.py
```
- **Preprocessing the dataset** For the preprocessing of captioning, we directly use the processed JSON blob from [neuraltalk](http://cs.stanford.edu/people/karpathy/deepimagesent/). Similar as step 2, set the PATH in *cnn_until.py* and *make_coco.py* to your own install path. Then :
```bash
$ cd data
$ python make_coco.py
```
- **Training**  The results are saved in the directory `exp`.
```bash
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sca_resnet_branch2b.py
```

## Citation

If you found this code useful, please cite the following paper:

> Chen L, Zhang H, Xiao J, et al. SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning[J]. arXiv preprint arXiv:1611.05594, 2016.

```
@article{chen2016sca,
  title={SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning},
  author={Chen, Long and Zhang, Hanwang and Xiao, Jun and Nie, Liqiang and Shao, Jian and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1611.05594},
  year={2016}
}

```
