import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/yaochuanqi/work/tmp/ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= 'ssdlite/coco/deploy.prototxt'
caffe_model='ssdlite/deploy.caffemodel'  
test_dir = "images"

caffe.set_mode_cpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

COCO_CLASSES = ("background" , "person" , "bicycle" , "car" , "motorcycle" , 
     "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light",
     "fire hydrant", "N/A" , "stop sign", "parking meter", "bench" , 
     "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , 
     "bear" , "zebra" , "giraffe" , "N/A" , "backpack" , "umbrella" , 
     "N/A" , "N/A" , "handbag" , "tie" , "suitcase" , "frisbee" , "skis" ,
     "snowboard" , "sports ball", "kite" , "baseball bat", "baseball glove",
     "skateboard" , "surfboard" , "tennis racket", "bottle" , "N/A" ,
     "wine glass", "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" ,
     "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog",
     "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant", 
     "bed" , "N/A" , "dining table", "N/A" , "N/A" , "toilet" , "N/A" ,
     "tv" , "laptop" , "mouse" , "remote" , "keyboard" , "cell phone",
     "microwave" , "oven" , "toaster" , "sink" , "refrigerator" , "N/A" ,
     "book" , "clock" , "vase" , "scissors" , "teddy bear", "hair drier",
     "toothbrush" )

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img / 127.5
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward() 
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (COCO_CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
