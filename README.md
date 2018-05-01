# MobileNetv2-SSDLite
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.

### Usage
0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
1. Use gen_model.py to generate the train.prototxt and deploy.prototxt (or use the default prototxt).
```
python gen_model.py -s train -l labelmap_coco.prototxt -d trainval_lmdb -c 91 >train.prototxt
python gen_model.py -s deploy -l labelmap_coco.prototxt -d trainval_lmdb -c 91 >deploy.prototxt
```
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.
3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.
4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.

### Note
There are some differences between caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer.

Under these circumstances, the detection result of converted model can not be very exact, so I will do some finetuning to recover the precision in the next few days.
