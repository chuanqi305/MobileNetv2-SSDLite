import tensorflow as tf
import cv2
import numpy as np
import os

def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'rb') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='',return_elements=[
              'image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])

#tensorflow: h, w, i, o
#            0, 1, 2, 3
#caffe:      o, i, h, w
#            3, 2, 0, 1
graph_create("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
if not os.path.exists('output'):
    os.mkdir('output')

with tf.Session() as sess:
    tensors = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
    for t in tensors:
      if t.name.endswith('weights') \
             or t.name.endswith('biases') \
             or t.name.endswith('moving_variance') \
             or t.name.endswith('moving_mean') \
             or t.name.endswith('beta') \
             or t.name.endswith('gamma'):
         ts = tf.get_default_graph().get_tensor_by_name(t.name + ":0")
         data = ts.eval()
         print(t.name)
         #print ts.get_shape()
         output_name = t.name.replace('FeatureExtractor/MobilenetV2/expanded_', '')
         output_name = output_name.replace('FeatureExtractor/MobilenetV2/', '')
         output_name = output_name.replace('BatchNorm', 'bn')
         output_name = output_name.replace('depthwise/depthwise', 'depthwise')
         output_name = output_name.replace('Conv2d_', '')
         output_name = output_name.replace('1x1_', '')
         output_name = output_name.replace('3x3_', '')
         output_name = output_name.replace('s2_', '')
         output_name = output_name.replace('_64', '')
         output_name = output_name.replace('_128', '')
         output_name = output_name.replace('_256', '')
         output_name = output_name.replace('_512', '')
         output_name = output_name.replace('/', '_')
         
         if len(data.shape) == 4:
             caffe_weights = data.transpose(3, 2, 0, 1)
             origin_shape = caffe_weights.shape
             boxes = 0
             if output_name.find('BoxEncodingPredictor') != -1:
                 boxes = caffe_weights.shape[0] // 4
             elif output_name.find('ClassPredictor') != -1:
                 boxes = caffe_weights.shape[0] // 91

             if output_name.find('BoxEncodingPredictor') != -1:
                 tmp = caffe_weights.reshape(boxes, 4, -1).copy()
                 new_weights = np.zeros(tmp.shape, dtype=np.float32)
                 #tf order:    [y, x, h, w]
                 #caffe order: [x, y, w, h]
                 if t.name == 'BoxPredictor_0/BoxEncodingPredictor/weights':
                     #caffe first box layer [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)]
                     #tf first box layer    [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)]
                     #adjust the box by weights and bias change
                     new_weights[:, 0] = tmp[:, 1] * 0.5
                     new_weights[:, 1] = tmp[:, 0] * 0.5
                 else:
                     new_weights[:, 0] = tmp[:, 1]
                     new_weights[:, 1] = tmp[:, 0]
                 new_weights[:, 2] = tmp[:, 3]
                 new_weights[:, 3] = tmp[:, 2]
                 caffe_weights = new_weights.reshape(origin_shape).copy()
             if output_name.find('BoxEncodingPredictor') != -1 or \
                 output_name.find('ClassPredictor') != -1:
                 tmp = caffe_weights.reshape(boxes, -1).copy()
                 new_weights = np.zeros(tmp.shape, dtype=np.float32)
                 #tf aspect ratio:   [1, 2, 3, 0.5, 0.333333333, 1]
                 #caffe aspect ratio:[1, 1, 2, 3, 0.5, 0.333333333]
                 if boxes == 6:
                     new_weights[0] = tmp[0]
                     new_weights[1] = tmp[5]
                     new_weights[2] = tmp[1]
                     new_weights[3] = tmp[2]
                     new_weights[4] = tmp[3]
                     new_weights[5] = tmp[4]
                     caffe_weights = new_weights.reshape(origin_shape).copy()
             caffe_weights.tofile('output/' + output_name + '.dat')
             print caffe_weights.shape
         else:
             caffe_bias = data
             boxes = 0
             if output_name.find('BoxEncodingPredictor') != -1:
                 boxes = caffe_bias.shape[0] // 4
             elif output_name.find('ClassPredictor') != -1:
                 boxes = caffe_bias.shape[0] // 91
             if output_name.find('BoxEncodingPredictor') != -1:
                 tmp = caffe_bias.reshape(boxes, 4).copy()
                 new_bias = np.zeros(tmp.shape, dtype=np.float32)
                 new_bias[:, 0] = tmp[:, 1]
                 new_bias[:, 1] = tmp[:, 0]
                 new_bias[:, 2] = tmp[:, 3]
                 new_bias[:, 3] = tmp[:, 2]
                 caffe_bias = new_bias.flatten().copy()

             if output_name.find('BoxEncodingPredictor') != -1 or \
                 output_name.find('ClassPredictor') != -1:
                 tmp = caffe_bias.reshape(boxes, -1).copy()
                 new_bias = np.zeros(tmp.shape, dtype=np.float32)
                 if boxes == 6:
                     new_bias[0] = tmp[0]
                     new_bias[1] = tmp[5]
                     new_bias[2] = tmp[1]
                     new_bias[3] = tmp[2]
                     new_bias[4] = tmp[3]
                     new_bias[5] = tmp[4]
                     print new_bias.shape
                     caffe_bias = new_bias.flatten()
                 elif t.name == 'BoxPredictor_0/BoxEncodingPredictor/biases':
                     #caffe first box layer [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)]
                     #tf first box layer    [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)]
                     #adjust the box by weights and bias change
                     new_bias[0,:2] = tmp[0,:2] * 0.5
                     new_bias[0,2] = tmp[0,2] + (np.log(0.5) / 0.2)
                     new_bias[0,3] = tmp[0,3] + (np.log(0.5) / 0.2)
                     new_bias[1] = tmp[1]
                     new_bias[2] = tmp[2]
                     caffe_bias = new_bias.flatten()
             caffe_bias.tofile('output/' + output_name + '.dat')

