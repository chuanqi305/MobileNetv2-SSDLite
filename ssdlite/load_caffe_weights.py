import numpy as np  
import sys,os  
caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

deploy_proto = 'deploy.prototxt'  
save_model = 'deploy.caffemodel'

weights_dir = 'output'
box_layers = ['conv_13/expand', 'Conv_1', 'layer_19_2_2', 'layer_19_2_3', 'layer_19_2_4', 'layer_19_2_5']
def load_weights(path, shape=None):
    weights = None
    if shape is None: 
        weights = np.fromfile(path, dtype=np.float32)
    else:
        weights = np.fromfile(path, dtype=np.float32).reshape(shape)
    os.unlink(path)
    return weights

def load_data(net):
    for key in net.params.iterkeys():
        if type(net.params[key]) is caffe._caffe.BlobVec:
            print key
            if 'mbox' not in key and (key.startswith("conv") or key.startswith("Conv") or key.startswith("layer")):
                print('conv')
                if key.endswith("/bn"):
                    prefix = weights_dir + '/' + key.replace('/', '_')
                    net.params[key][0].data[...] = load_weights(prefix + '_moving_mean.dat')
                    net.params[key][1].data[...] = load_weights(prefix + '_moving_variance.dat')
                    net.params[key][2].data[...] = np.ones(net.params[key][2].data.shape, dtype=np.float32)
                elif key.endswith("/scale"):
                    prefix = weights_dir + '/' + key.replace('scale','bn').replace('/', '_')
                    net.params[key][0].data[...] = load_weights(prefix + '_gamma.dat')
                    net.params[key][1].data[...] = load_weights(prefix + '_beta.dat')
                else:
                    prefix = weights_dir + '/' + key.replace('/', '_')
                    ws = np.ones((net.params[key][0].data.shape[0], 1, 1, 1), dtype=np.float32)
                    if os.path.exists(prefix + '_weights_scale.dat'):
                        ws = load_weights(prefix + '_weights_scale.dat', ws.shape)
                    net.params[key][0].data[...] = load_weights(prefix + '_weights.dat', net.params[key][0].data.shape) * ws
                    if len(net.params[key]) > 1:
                        net.params[key][1].data[...] = load_weights(prefix + '_biases.dat')
                        
            elif 'mbox_loc/depthwise' in key or 'mbox_conf/depthwise' in key:
                prefix = key[0:key.find('_mbox')]
                index = box_layers.index(prefix)
                if 'mbox_loc' in key:
                    prefix = weights_dir + '/BoxPredictor_' + str(index) + '_BoxEncodingPredictor_depthwise'
                else:
                    prefix = weights_dir + '/BoxPredictor_' + str(index) + '_ClassPredictor_depthwise'
                if key.endswith("/bn"):
                    net.params[key][0].data[...] = load_weights(prefix + '_bn_moving_mean.dat')
                    net.params[key][1].data[...] = load_weights(prefix + '_bn_moving_variance.dat')
                    net.params[key][2].data[...] = np.ones(net.params[key][2].data.shape, dtype=np.float32)
                elif key.endswith("/scale"):
                    net.params[key][0].data[...] = load_weights(prefix + '_gamma.dat')
                    net.params[key][1].data[...] = load_weights(prefix + '_beta.dat')
                else:
                    print key
                    net.params[key][0].data[...] = load_weights(prefix + '_weights.dat', net.params[key][0].data.shape)
                    if len(net.params[key]) > 1:
                        net.params[key][1].data[...] = load_weights(prefix + '_biases.dat')
            elif key.endswith("mbox_loc"):
                prefix = key.replace("_mbox_loc", "")
                index = box_layers.index(prefix)
                prefix = weights_dir + '/BoxPredictor_' + str(index) + '_BoxEncodingPredictor'
                net.params[key][0].data[...] = load_weights(prefix + '_weights.dat', net.params[key][0].data.shape)
                net.params[key][1].data[...] = load_weights(prefix + '_biases.dat')
            elif key.endswith("mbox_conf"):
                prefix = key.replace("_mbox_conf", "")
                index = box_layers.index(prefix)
                prefix = weights_dir + '/BoxPredictor_' + str(index) + '_ClassPredictor'
                net.params[key][0].data[...] = load_weights(prefix + '_weights.dat', net.params[key][0].data.shape)
                net.params[key][1].data[...] = load_weights(prefix + '_biases.dat')
            else:
                print ("error key " + key)
                  
  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

load_data(net_deploy)
net_deploy.save(save_model)

