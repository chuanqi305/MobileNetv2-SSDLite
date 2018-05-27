import tensorflow as tf

def graph_create(graphpath):
    with tf.gfile.FastGFile(graphpath, 'r') as graphfile:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(graphfile.read())

        return tf.import_graph_def(graphdef, name='',return_elements=[])
graph_create("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
with tf.Session() as sess:
    tf.summary.FileWriter("log/",sess.graph)
    tensors = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
    for t in tensors:
      if t.name.endswith('weights'):
         ts = tf.get_default_graph().get_tensor_by_name(t.name + ":0")
         print(t.name)
         print ts.get_shape()
