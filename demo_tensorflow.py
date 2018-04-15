import tensorflow as tf
import cv2
import numpy as np

def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'r') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='',return_elements=[
              'image_tensor:0','detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])

image_tensor,  box, score, cls = graph_create("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
image_file = "images/004545.jpg"
with tf.Session() as sess:
     image = cv2.imread(image_file)
     image_data = np.expand_dims(image, axis=0).astype(np.uint8)

     b, s, c = sess.run([box, score, cls], {image_tensor: image_data})
     boxes = b[0]
     conf = s[0]
     clses = c[0]
     #writer = tf.summary.FileWriter('debug', sess.graph)

     for i in range(8):
         bx = boxes[i]
         print boxes[i]
         print conf[i]
         print clses[i]
         if conf[i] < 0.5:
             continue
         h = image.shape[0]
         w = image.shape[1]
         p1 = (int(w * bx[1]), int(h * bx[0]))
         p2 = (int(w * bx[3]) ,int(h * bx[2]))
         cv2.rectangle(image, p1, p2, (0,255,0))

     cv2.imshow("mobilenet-ssd", image)
     cv2.waitKey(0) 
