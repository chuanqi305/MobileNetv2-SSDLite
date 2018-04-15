import tensorflow as tf
import cv2
import numpy as np

def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'r') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='',return_elements=[
              'image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])

#tensorflow: h, w, i, o
#            0, 1, 2, 3
#caffe:      o, i, h, w
#            3, 2, 0, 1
graph_create("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
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
         print ts.get_shape()
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
             data.transpose(3, 2, 0, 1).tofile('output/' + output_name + '.dat')
         else:
             data.tofile('output/' + output_name + '.dat')

