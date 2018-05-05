import numpy as np  
import sys,os  
from scipy import misc
import cv2
caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

def load_net(net, net2):
    #maps = [0,5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
    maps = [0,5,2,16,9,44,6,3,17,62,21,67,18,19,4,1,64,20,63,7,72]
    for key in net.params.iterkeys():
        if type(net.params[key]) is not caffe._caffe.BlobVec:
            break
        else:
            if key.endswith('mbox_conf'):
                for i in range(len(net.params[key])):
                    wt = net.params[key][i].data
                    x = wt.shape[0] / 91
                    wt = wt.reshape(x, 91, -1)
                    neww = np.ones((x, 21, wt.shape[2]))
                    print(neww.shape)
                    print(wt.shape)
                    for j,m in enumerate(maps):
                       neww[:,j,:] = wt[:,m,:]
                    net2.params[key][i].data[...] = neww.reshape(net2.params[key][i].data.shape)
                print(key)
               
            else:
                for i in range(len(net.params[key])):
                    net2.params[key][i].data[...] = net.params[key][i].data
  
from_file = "deploy.prototxt"
from_model = "deploy.caffemodel"

net_file= 'voc/deploy.prototxt'  
save_model='deploy_voc.caffemodel'

net = caffe.Net(from_file,from_model,caffe.TEST)  
net2 = caffe.Net(net_file,caffe.TRAIN)  

load_net(net, net2)
net2.save(save_model)

