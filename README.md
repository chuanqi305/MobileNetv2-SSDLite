# MobileNetv2-SSDLite
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.

### Introduction
This is a MobileNetv2-SSDLite model converted from tensorflow, currently only ".prototxt" file
is available.

### Usage
0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
1. Use gen_model.py to generate the train.prototxt and deploy.prototxt.
```
python gen_model.py -s train -l labelmap_coco.prototxt -d trainval_lmdb -c 91 >train.prototxt
python gen_model.py -s deploy -l labelmap_coco.prototxt -d trainval_lmdb -c 91 >deploy.prototxt
```
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.
3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.
4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.

### Note
Currently the converted caffemodel is not available. Wait patiently, please.

