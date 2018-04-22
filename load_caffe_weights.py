import numpy as np  
import sys,os  
caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

deploy_proto = 'deploy.prototxt'  
save_model = 'deploy.caffemodel'

box_layers = ['conv_13/expand', 'Conv_1', 'layer_19_2_2', 'layer_19_2_3', 'layer_19_2_4', 'layer_19_2_5']

def load_data(net):
    for key in net.params.iterkeys():
        if type(net.params[key]) is caffe._caffe.BlobVec:
            print key
            if key.find('mbox') == -1 and (key.startswith("conv") or key.startswith("Conv") or key.startswith("layer")):
                print('conv')
                if key.endswith("/bn"):
                    prefix = 'output/' + key.replace('/', '_')
                    net.params[key][0].data[...] = np.fromfile(prefix + '_moving_mean.dat', dtype=np.float32)
                    net.params[key][1].data[...] = np.fromfile(prefix + '_moving_variance.dat', dtype=np.float32)
                    net.params[key][2].data[...] = np.ones(net.params[key][2].data.shape, dtype=np.float32)
                elif key.endswith("/scale"):
                    prefix = 'output/' + key.replace('scale','bn').replace('/', '_')
                    net.params[key][0].data[...] = np.fromfile(prefix + '_gamma.dat', dtype=np.float32)
                    net.params[key][1].data[...] = np.fromfile(prefix + '_beta.dat', dtype=np.float32)
                else:
                    prefix = 'output/' + key.replace('/', '_')
                    net.params[key][0].data[...] = np.fromfile(prefix + '_weights.dat', dtype=np.float32).reshape(net.params[key][0].data.shape)
                    if len(net.params[key]) > 1:
                        net.params[key][1].data[...] = np.fromfile(prefix + '_biases.dat', dtype=np.float32)
            elif key.endswith("mbox_loc"):
                prefix = key.replace("_mbox_loc", "")
                index = box_layers.index(prefix)
                prefix = 'output/BoxPredictor_' + str(index) + '_BoxEncodingPredictor'
                net.params[key][0].data[...] = np.fromfile(prefix + '_weights.dat', dtype=np.float32).reshape(net.params[key][0].data.shape)
                net.params[key][1].data[...] = np.fromfile(prefix + '_biases.dat', dtype=np.float32)
            elif key.endswith("mbox_conf"):
                prefix = key.replace("_mbox_conf", "")
                index = box_layers.index(prefix)
                prefix = 'output/BoxPredictor_' + str(index) + '_ClassPredictor'
                net.params[key][0].data[...] = np.fromfile(prefix + '_weights.dat', dtype=np.float32).reshape(net.params[key][0].data.shape)
                net.params[key][1].data[...] = np.fromfile(prefix + '_biases.dat', dtype=np.float32)
            else:
                print ("error key " + key)
                  
  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

load_data(net_deploy)
net_deploy.save(save_model)

