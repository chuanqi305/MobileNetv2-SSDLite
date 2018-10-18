import os
import sys
import argparse
import logging
import math

try:
    caffe_root = '/home/yaochuanqi/work/ssd/caffe/build2/install/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    logging.fatal("Cannot find caffe!")
from google.protobuf import text_format

class CaffeNetGenerator:
    def __init__(self, net):
        self.net = net
        self.top = "data"
        self.eps = 0.001
        self.first_prior = True
        self.anchors = create_ssd_anchors()
        self.shape = {}

    def header(self, name):
        self.net.name = name

    def data_deploy(self):
        self.net.input.append("data")
        shape = self.net.input_shape.add()
        shape.dim.append(1) 
        shape.dim.append(3) 
        shape.dim.append(self.input_size) 
        shape.dim.append(self.input_size) 

    def data_train_classifier(self):
        layer = self.net.layer.add()
        layer.name = "data"
        layer.type = "Data"
        layer.top.append("data")
        layer.top.append("label")
        layer.data_param.source = self.lmdb
        layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        layer.data_param.batch_size = 64
        layer.transform_param.crop_size = self.input_size
        layer.transform_param.mean_file = "imagenet.mean"
        layer.transform_param.mirror = True
        layer.include.add().phase = caffe_pb2.TRAIN

    def data_train_ssd(self):
        layer = self.net.layer.add()
        layer.name = "data"
        layer.type = "AnnotatedData"
        layer.top.append("data")
        layer.top.append("label")
        layer.include.add().phase = caffe_pb2.TRAIN

        layer.transform_param.scale = 0.007843
        layer.transform_param.mirror = True
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.resize_param.prob = 1.0
        layer.transform_param.resize_param.resize_mode = caffe_pb2.ResizeParameter.WARP
        layer.transform_param.resize_param.height = self.input_size
        layer.transform_param.resize_param.width = self.input_size
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.LINEAR)
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.AREA)
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.NEAREST)
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.CUBIC)
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.LANCZOS4)
        layer.transform_param.emit_constraint.emit_type = caffe_pb2.EmitConstraint.CENTER
        layer.transform_param.distort_param.brightness_prob = 0.5
        layer.transform_param.distort_param.brightness_delta = 32.0
        layer.transform_param.distort_param.contrast_lower = 0.5
        layer.transform_param.distort_param.contrast_upper = 1.5
        layer.transform_param.distort_param.hue_prob = 0.5
        layer.transform_param.distort_param.hue_delta = 18.0
        layer.transform_param.distort_param.saturation_prob = 0.5
        layer.transform_param.distort_param.saturation_lower = 0.5
        layer.transform_param.distort_param.saturation_upper = 1.5
        layer.transform_param.distort_param.random_order_prob = 0.0
        layer.transform_param.expand_param.prob = 0.5
        layer.transform_param.expand_param.max_expand_ratio = 4.0


        layer.data_param.source = self.lmdb
        layer.data_param.batch_size = 64
        layer.data_param.backend = caffe_pb2.DataParameter.LMDB

        sampler = layer.annotated_data_param.batch_sampler.add()
        sampler.max_sample = 1
        sampler.max_trials = 1
        for overlap in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            sampler = layer.annotated_data_param.batch_sampler.add()
            sampler.sampler.min_scale = 0.3
            sampler.sampler.max_scale = 1.0
            sampler.sampler.min_aspect_ratio = 0.5
            sampler.sampler.max_aspect_ratio = 2.0
            sampler.sample_constraint.min_jaccard_overlap = overlap
            sampler.max_sample = 1
            sampler.max_trials = 50
        layer.annotated_data_param.label_map_file = self.label_map

    def data_test_ssd(self):
        layer = self.net.layer.add()
        layer.name = "data"
        layer.type = "AnnotatedData"
        layer.top.append("data")
        layer.top.append("label")
        layer.include.add().phase = caffe_pb2.TEST
        layer.transform_param.scale = 0.007843
        layer.transform_param.mirror = True
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.mean_value.append(127.5)
        layer.transform_param.resize_param.prob = 1.0
        layer.transform_param.resize_param.resize_mode = caffe_pb2.ResizeParameter.WARP
        layer.transform_param.resize_param.height = self.input_size
        layer.transform_param.resize_param.width = self.input_size
        layer.transform_param.resize_param.interp_mode.append(caffe_pb2.ResizeParameter.LINEAR)

        layer.data_param.source = ""
        layer.data_param.batch_size = 8
        layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        layer.annotated_data_param.label_map_file = self.label_map

    def classifier_loss(self):
        layer = self.net.layer.add()
        layer.name = "softmax"
        layer.type = "Softmax"
        layer.bottom.append(self.top)
        layer.top.append("prob")

        layer = self.net.layer.add()
        layer.name = "accuracy"
        layer.type = "Accuracy"
        layer.bottom.append("prob")
        layer.bottom.append("label")
        layer.top.append("accuracy")

        layer = self.net.layer.add()
        layer.name = "loss"
        layer.type = "SoftmaxWithLoss"
        layer.bottom.append(self.top)
        layer.bottom.append("label")
        layer.loss_weight.append(1)

    def ssd_predict(self):
        layer = self.net.layer.add()
        layer.name = "mbox_conf_reshape"
        layer.type = "Reshape"
        layer.bottom.append("mbox_conf")
        layer.top.append("mbox_conf_reshape")
        layer.reshape_param.shape.dim.append(0)
        layer.reshape_param.shape.dim.append(-1)
        layer.reshape_param.shape.dim.append(self.class_num)

        layer = self.net.layer.add()
        layer.name = "mbox_conf_sigmoid"
        layer.type = "Sigmoid"
        layer.bottom.append("mbox_conf_reshape")
        layer.top.append("mbox_conf_sigmoid")
        layer = self.net.layer.add()
        layer.name = "mbox_conf_flatten"
        layer.type = "Flatten"
        layer.bottom.append("mbox_conf_sigmoid")
        layer.top.append("mbox_conf_flatten")
        layer.flatten_param.axis = 1

        layer = self.net.layer.add()
        layer.name = "detection_out"
        layer.type = "DetectionOutput"
        layer.bottom.append("mbox_loc")
        layer.bottom.append("mbox_conf_flatten")
        layer.bottom.append("mbox_priorbox")
        layer.top.append("detection_out")
        layer.include.add().phase = caffe_pb2.TEST

        layer.detection_output_param.num_classes = self.class_num
        layer.detection_output_param.share_location = True
        layer.detection_output_param.background_label_id = 0
        layer.detection_output_param.nms_param.nms_threshold = 0.45
        layer.detection_output_param.nms_param.top_k = 100
        layer.detection_output_param.code_type = caffe_pb2.PriorBoxParameter.CENTER_SIZE
        layer.detection_output_param.keep_top_k = 100
        layer.detection_output_param.confidence_threshold = 0.35

    def ssd_test(self):
        self.ssd_predict()
        layer = self.net.layer.add() 
        layer.name = "detection_eval"
        layer.type = "DetectionEvaluate"
        layer.bottom.append("detection_out")
        layer.bottom.append("label")
        layer.top.append("detection_eval")
        layer.include.add().phase = caffe_pb2.TEST
        layer.detection_evaluate_param.num_classes = self.class_num
        layer.detection_evaluate_param.background_label_id = 0
        layer.detection_evaluate_param.overlap_threshold = 0.5
        layer.detection_evaluate_param.evaluate_difficult_gt = False

    def ssd_loss(self):
        layer = self.net.layer.add() 
        layer.name = "mbox_loss"
        layer.type = "MultiBoxLoss"
        layer.bottom.append("mbox_loc")
        layer.bottom.append("mbox_conf")
        layer.bottom.append("mbox_priorbox")
        layer.bottom.append("label")
        layer.top.append("mbox_loss")
        layer.include.add().phase = caffe_pb2.TRAIN
        layer.propagate_down.append(True)
        layer.propagate_down.append(True)
        layer.propagate_down.append(False)
        layer.propagate_down.append(False)
        layer.loss_param.normalization = caffe_pb2.LossParameter.VALID
        layer.multibox_loss_param.loc_loss_type = caffe_pb2.MultiBoxLossParameter.SMOOTH_L1
        layer.multibox_loss_param.conf_loss_type = caffe_pb2.MultiBoxLossParameter.LOGISTIC
        layer.multibox_loss_param.loc_weight = 1.0
        layer.multibox_loss_param.num_classes = self.class_num
        layer.multibox_loss_param.share_location = True
        layer.multibox_loss_param.match_type = caffe_pb2.MultiBoxLossParameter.PER_PREDICTION
        layer.multibox_loss_param.overlap_threshold = 0.5
        layer.multibox_loss_param.use_difficult_gt = True
        layer.multibox_loss_param.neg_pos_ratio = 3.0
        layer.multibox_loss_param.neg_overlap = 0.5
        layer.multibox_loss_param.code_type = caffe_pb2.PriorBoxParameter.CENTER_SIZE
        layer.multibox_loss_param.ignore_cross_boundary_bbox = False
        layer.multibox_loss_param.mining_type = caffe_pb2.MultiBoxLossParameter.MAX_NEGATIVE

    def concat_boxes(self, convs):
        for lc in ["loc", "conf"]:
            layer = self.net.layer.add()
            layer.name = "mbox_" + lc
            layer.type = "Concat"
            for conv in convs:
                layer.bottom.append(conv + "_mbox_" + lc + "_flat")
            layer.top.append("mbox_" + lc)
            layer.concat_param.axis = 1
        layer = self.net.layer.add()
        layer.name = "mbox_priorbox"
        layer.type = "Concat"
        for conv in convs:
            layer.bottom.append(conv + "_mbox_priorbox")
        layer.top.append("mbox_priorbox")
        layer.concat_param.axis = 2

    def adjust_pad(self):
       '''
       simulate tensorflow padding with caffe slice layer
       '''
       name = self.net.layer[-1].top[0]
       self.net.layer[-1].top[0] = name + "/pad"
       self.net.layer[-1].convolution_param.pad[0] += 1
       layer = self.net.layer.add()
       layer.name = "slice"
       layer.type = "Slice"
       layer.bottom.append(name + "/pad")
       layer.top.append(name + "/margin1")
       layer.top.append(name + "/tmp")
       layer.slice_param.axis = 2
       layer.slice_param.slice_point.append(1)
       layer = self.net.layer.add()
       layer.name = "slice"
       layer.type = "Slice"
       layer.bottom.append(name + "/tmp")
       layer.top.append(name + "/margin2")
       layer.top.append(name)
       layer.slice_param.axis = 3
       layer.slice_param.slice_point.append(1)
       self.need_silence_layer.append(name + "/margin1")
       self.need_silence_layer.append(name + "/margin2")
    
    def conv(self, name, output, kernel, stride=1, group=1, bias=False, bottom=None):
        layer = self.net.layer.add()
        layer.name = name
        if bottom is None:
            bottom = self.top
        layer.bottom.append(bottom)
        layer.type = "Convolution"
        layer.top.append(name)
        layer.convolution_param.num_output = output
        lr_decay_mult = [[1.0, 1.0], [2,0, 0.0]]
        #print name + "->" + str(bias)
        if self.nobn:
            bias = True
        if not bias:
            layer.convolution_param.bias_term = bias
            lr_decay_mult = [[1.0, 1.0]]
        for mul in lr_decay_mult:
            param = layer.param.add()
            param.lr_mult = mul[0]
            param.decay_mult = mul[1]
        if group > 1:
            layer.convolution_param.group = group
        if kernel > 1:
            layer.convolution_param.pad.append(kernel / 2)
        if stride > 1:
            layer.convolution_param.stride.append(stride)
        layer.convolution_param.kernel_size.append(kernel)
        layer.convolution_param.weight_filler.type = "msra"
        if bias:
            layer.convolution_param.bias_filler.type = "constant"
            layer.convolution_param.bias_filler.value = 0
        n, c, h, w = self.shape[bottom]
        pad, output_size = compute_pad((h, w), stride)
        self.top = name
        self.shape[name] = (1, output) + output_size
        if self.tfpad:
            if pad[0] != pad[2] or pad[1] != pad[3]:
                self.adjust_pad()

    def bn(self, name): 
        if self.nobn:
            return
        layer = self.net.layer.add()
        layer.name = "%s/bn" % name
        layer.type = "BatchNorm"
        layer.bottom.append(name)
        layer.top.append(name)
        for i in range(3):
            param = layer.param.add()
            param.lr_mult = 0
            param.decay_mult = 0
        if self.eps != 1e-5:
            layer.batch_norm_param.eps = self.eps
        layer = self.net.layer.add()
        layer.name = "%s/scale" % name
        layer.type = "Scale"
        layer.bottom.append(name)
        layer.top.append(name)
        for mul in [[1.0, 0.0], [2.0, 0.0]]:
            param = layer.param.add()
            param.lr_mult = mul[0]
            param.decay_mult = mul[1]
        layer.scale_param.filler.value = 1
        layer.scale_param.bias_term = True
        layer.scale_param.bias_filler.value = 0

    def relu(self, name):
        layer = self.net.layer.add()
        layer.name = "%s/relu" % name
        if self.relu6:
            layer.type = "ReLU6"
        else:
            layer.type = "ReLU"
        layer.bottom.append(name)
        layer.top.append(name)

    def shortcut(self, bottom, top):
        layer = self.net.layer.add()
        layer.name = top + "/sum"
        layer.type = "Eltwise"
        layer.bottom.append(bottom)
        layer.bottom.append(self.top)
        layer.top.append(top)
        self.top = top
        self.shape[top] = self.shape[bottom]

    def ave_pool(self, name):
        layer = self.net.layer.add()
        layer.name = name
        layer.type = "Pooling"
        layer.bottom.append(self.top)
        layer.top.append(name)
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE 
        layer.pooling_param.global_pooling = True
        self.top = name

    def permute(self, name):
        layer = self.net.layer.add()
        layer.name = "%s_perm" % name
        layer.type = "Permute"
        layer.bottom.append(name)
        layer.top.append("%s_perm" % name)
        for i in [0, 2, 3, 1]:
            layer.permute_param.order.append(i)
        self.top = name + "_perm"

    def flatten(self, name):
        layer = self.net.layer.add()
        layer.name = "%s_flat" % name
        layer.type = "Flatten"
        layer.bottom.append(name + "_perm")
        layer.top.append("%s_flat" % name)
        layer.flatten_param.axis = 1
        self.top = name + "_flat"

    def mbox_prior(self, name, min_size, max_size, aspect_ratio):
        min_box = self.input_size * min_size
        layer = self.net.layer.add()
        layer.name = "%s_mbox_priorbox" % name
        layer.type = "PriorBox"
        layer.bottom.append(name)
        layer.bottom.append("data")
        layer.top.append("%s_mbox_priorbox" % name)
        layer.prior_box_param.min_size.append(float(min_box))
        if max_size is not None:
            max_box = self.input_size * max_size
            layer.prior_box_param.max_size.append(max_box)
        for ar in aspect_ratio:
            layer.prior_box_param.aspect_ratio.append(ar)

        layer.prior_box_param.flip = True
        layer.prior_box_param.clip = False
        for i in [0.1, 0.1, 0.2, 0.2]:
            layer.prior_box_param.variance.append(i)
        layer.prior_box_param.offset = 0.5

    def fc(self, name, output): 
        layer = self.net.layer.add()
        layer.name = name
        layer.type = "InnerProduct"
        layer.bottom.append(self.top)
        layer.top.append(name)
        for i in [[1, 1], [2, 0]]:
            param = layer.param.add()
            param.lr_mult = i[0]
            param.decay_mult = i[1]
        layer.inner_product_param.num_output = output
        layer.weight_filler.type = "msra"
        layer.bias_filler.type = "constant"
        layer.bias_filler.value = 0
 
    def reshape(self, name, output):    
        layer = self.net.layer.add()
        layer.name = name
        layer.type = "Reshape"
        layer.bottom.append(self.top)
        layer.top.append(name)
        for i in [-1, output, 1, 1]:
            layer.reshape_param.shape.dim.append(i)

    def silence(self):
        if len(self.need_silence_layer) == 0:
            return
        layer = self.net.layer.add()
        layer.name = "silence"
        layer.type = "Silence"
        for bottom in self.need_silence_layer:
            layer.bottom.append(bottom)

    def conv_bn_relu(self, name, num, kernel, stride):
      self.conv(name, num, kernel, stride)
      self.bn(name)
      self.relu(name)

    def conv_bn_relu_with_factor(self, name, outp, kernel, stride):
      outp = int(outp * self.size)
      self.conv(name, outp, kernel, stride)
      self.bn(name)
      self.relu(name)

    def conv_ssd(self, name, stage, inp, outp):
      stage = str(stage)
      self.conv_expand(name + '_1_' + stage, inp, outp / 2)
      self.conv_depthwise(name + '_2_' + stage + '/depthwise', outp / 2, 2)
      self.conv_expand(name + '_2_' + stage, outp / 2, outp)

    def conv_block(self, name, inp, t, outp, stride, sc):
      last_block = self.top
      self.conv_expand(name + '/expand', inp, t * inp)
      self.conv_depthwise(name + '/depthwise', t * inp, stride)
      if sc:
         self.conv_project(name + '/project', t * inp, outp)
         self.shortcut(last_block, name)
      else:
         self.conv_project(name + '/project', t * inp, outp)
    
    def conv_depthwise(self, name, inp, stride, bottom=None):
      inp = int(inp * self.size)
      self.conv(name, inp, 3, stride, inp, bottom=bottom)
      self.bn(name)
      self.relu(name)

    def conv_expand(self, name, inp, outp):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      self.conv(name, outp, 1)
      self.bn(name)
      self.relu(name)

    def conv_project(self, name, inp, outp):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      self.conv(name, outp, 1)
      self.bn(name)

    def conv_dw_pw(self, name, inp, outp, stride):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      name1 = name + "/depthwise"
      self.conv(name1, inp, 3, stride, inp)
      self.bn(name1)
      self.relu(name1)
      name2 = name 
      self.conv(name2, outp, 1)
      self.bn(name2)
      self.relu(name2)

    def mbox_conf_ssdlite(self, bottom, inp, num):
       name = bottom + "_mbox_conf"
       self.conv_depthwise(name + '/depthwise', inp, 1, bottom=bottom)
       self.conv(name, num, 1, bias=True)
       self.permute(name)
       self.flatten(name)

    def mbox_loc_ssdlite(self, bottom, inp, num):
       name = bottom + "_mbox_loc"
       self.conv_depthwise(name + '/depthwise', inp, 1, bottom=bottom)
       self.conv(name, num, 1, bias=True)
       self.permute(name)
       self.flatten(name)

    def mbox_ssdlite(self, bottom, num):
       inp = self.shape[bottom][1]
       self.mbox_loc_ssdlite(bottom, inp, num * 4)
       self.mbox_conf_ssdlite(bottom, inp, num * self.class_num)
       min_size, max_size = self.anchors[0]
       if self.first_prior:
           self.mbox_prior(bottom, min_size, None, [2.0])
           self.first_prior = False
       else:
           self.mbox_prior(bottom, min_size, max_size, [2.0,3.0])
       self.anchors.pop(0)

    def mbox_conf_ssd(self, bottom, num):
       name = bottom + "_mbox_conf"
       self.conv(name, num, 3, bias=True, bottom=bottom)
       self.permute(name)
       self.flatten(name)

    def mbox_loc_ssd(self, bottom, num):
       name = bottom + "_mbox_loc"
       self.conv(name, num, 3, bias=True, bottom=bottom)
       self.permute(name)
       self.flatten(name)

    def mbox_ssd(self, bottom, num):
       self.mbox_loc_ssd(bottom, num * 4)
       self.mbox_conf_ssd(bottom, num * self.class_num)
       min_size, max_size = self.anchors[0]
       if self.first_prior:
           self.mbox_prior(bottom, min_size, None, [2.0])
           self.first_prior = False
       else:
           self.mbox_prior(bottom, min_size, max_size, [2.0,3.0])
       self.anchors.pop(0)

    def mbox(self, bottom, num):
       if self.islite:
           self.mbox_ssdlite(bottom, num)
       else:
           self.mbox_ssd(bottom, num)

    def init(self, FLAGS):
        self.class_num = FLAGS.class_num
        self.lmdb = FLAGS.lmdb
        self.stage = FLAGS.stage
        if FLAGS.lmdb == "":
            if self.stage == "train":
                self.lmdb = "trainval_lmdb"
            elif self.stage == "test":
                self.lmdb = "test_lmdb"
        self.label_map = FLAGS.label_map
        self.eps = FLAGS.eps
        self.relu6 = FLAGS.relu6
        self.nobn = FLAGS.nobn
        self.tfpad = FLAGS.tfpad
        self.gen_ssd = not FLAGS.classifier
        if self.gen_ssd:
            self.input_size = 300
        else:
            self.input_size = 224
        self.size = FLAGS.size
        if FLAGS.type == "ssdlite":
            self.islite = True
        else:
            self.islite = False
        self.shape["data"] = (1, 3, self.input_size, self.input_size)
        self.need_silence_layer = []

    def gen_mobile_ssd(self):
        if self.gen_ssd:
            self.header("MobileNet-SSD")
        else:
            self.header("MobileNet")
        if self.stage == "train":
            if gen_ssd:
                assert(self.lmdb is not None)
                assert(self.label_map is not None)
                self.data_train_ssd()
            else:
                assert(self.lmdb is not None)
                self.data_train_classifier()
        elif self.stage == "test":
            self.data_test_ssd()
        else:
            self.data_deploy()
        self.conv_bn_relu_with_factor("conv0", 32, 3, 2)
        self.conv_dw_pw("conv1", 32,  64, 1)
        self.conv_dw_pw("conv2", 64, 128, 2)
        self.conv_dw_pw("conv3", 128, 128, 1)
        self.conv_dw_pw("conv4", 128, 256, 2)
        self.conv_dw_pw("conv5", 256, 256, 1)
        self.conv_dw_pw("conv6", 256, 512, 2) 
        self.conv_dw_pw("conv7", 512, 512, 1)
        self.conv_dw_pw("conv8", 512, 512, 1)
        self.conv_dw_pw("conv9", 512, 512, 1)
        self.conv_dw_pw("conv10",512, 512, 1)
        self.conv_dw_pw("conv11",512, 512, 1)
        self.conv_dw_pw("conv12",512, 1024, 2) 
        self.conv_dw_pw("conv13",1024, 1024, 1) 
        if self.gen_ssd:
            self.conv_bn_relu("conv14_1", 256, 1, 1)
            self.conv_bn_relu("conv14_2", 512, 3, 2)
            self.conv_bn_relu("conv15_1", 128, 1, 1)
            self.conv_bn_relu("conv15_2", 256, 3, 2)
            self.conv_bn_relu("conv16_1", 128, 1, 1)
            self.conv_bn_relu("conv16_2", 256, 3, 2)
            self.conv_bn_relu("conv17_1", 64,  1, 1)
            self.conv_bn_relu("conv17_2", 128, 3, 2)
            self.mbox("conv11", 3)
            self.mbox("conv13", 6)
            self.mbox("conv14_2", 6)
            self.mbox("conv15_2", 6)
            self.mbox("conv16_2", 6)
            self.mbox("conv17_2", 6)
            self.concat_boxes(['conv11', 'conv13', 'conv14_2', 'conv15_2', 'conv16_2', 'conv17_2'])
            if self.stage == "train":
               self.ssd_loss()
            elif self.stage == "deploy":
               self.ssd_predict()
            else:
               self.ssd_test()
        else:
            self.ave_pool("pool")
            self.conv("fc", self.class_num, 1, 1, 1, True)
            if self.stage == "train":
               self.classifier_loss()

    def gen_mobilev2_ssd(self):
        if self.gen_ssd:
            self.header("MobileNetv2-SSDLite")
        else:
            self.header("MobileNetv2")
        if self.stage == "train":
            if self.gen_ssd:
                assert(self.lmdb is not None)
                assert(self.label_map is not None)
                self.data_train_ssd()
            else:
                assert(self.lmdb is not None)
                self.data_train_classifier()
        elif self.stage == "test":
            self.data_test_ssd()
        else:
            self.data_deploy()
        self.conv_bn_relu_with_factor("Conv", 32, 3, 2)
        self.conv_depthwise("conv/depthwise", 32, 1)
        self.conv_project("conv/project", 32, 16)
        self.conv_block("conv_1", 16,  6, 24, 2, False)
        self.conv_block("conv_2", 24,  6, 24, 1, True)
        self.conv_block("conv_3", 24,  6, 32, 2, False)
        self.conv_block("conv_4", 32,  6, 32, 1, True)
        self.conv_block("conv_5", 32,  6, 32, 1, True)
        self.conv_block("conv_6", 32,  6, 64, 2, False)
        self.conv_block("conv_7", 64,  6, 64, 1, True)
        self.conv_block("conv_8", 64,  6, 64, 1, True)
        self.conv_block("conv_9", 64,  6, 64, 1, True)
        self.conv_block("conv_10", 64,  6, 96, 1, False)
        self.conv_block("conv_11", 96,  6, 96, 1, True)
        self.conv_block("conv_12", 96,  6, 96, 1, True)
        self.conv_block("conv_13", 96,  6, 160, 2, False)
        self.conv_block("conv_14", 160,  6, 160, 1, True)
        self.conv_block("conv_15", 160,  6, 160, 1, True)
        self.conv_block("conv_16", 160,  6, 320, 1, False)
        self.conv_bn_relu_with_factor("Conv_1", 1280, 1, 1)
        if self.gen_ssd is True:
            self.conv_ssd("layer_19", 2, 1280, 512)
            self.conv_ssd("layer_19", 3, 512, 256)
            self.conv_ssd("layer_19", 4, 256, 256)
            self.conv_ssd("layer_19", 5, 256, 128)
            self.silence()
            self.nobn = False
            self.mbox("conv_13/expand", 3)
            self.mbox("Conv_1", 6)
            self.mbox("layer_19_2_2", 6)
            self.mbox("layer_19_2_3", 6)
            self.mbox("layer_19_2_4", 6)
            self.mbox("layer_19_2_5", 6)
            self.concat_boxes(['conv_13/expand', 'Conv_1', 'layer_19_2_2', 'layer_19_2_3', 'layer_19_2_4', 'layer_19_2_5'])
            if self.stage == "train":
               self.ssd_loss()
            elif self.stage == "deploy":
               self.ssd_predict()
            else:
               self.ssd_test()
        else:
            self.ave_pool("pool")
            self.conv("fc", self.class_num, 1, 1, 1, True)
            if self.stage == "train":
               self.classifier_loss()

    def generate(self, FLAGS):
        self.init(FLAGS)
        if FLAGS.version == "1":
            self.gen_mobile_ssd()
        elif FLAGS.version == "2":
            self.gen_mobilev2_ssd()
        else:
            print "version " + FLAGS.version + " is not supported"
            exit(-1)
   
def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95):
    box_specs_list = []
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
    return zip(scales[:-1], scales[1:])

def compute_pad(inp, stride, tf=True):
    H = inp[0]
    W = inp[1]
    S = stride
    F = 3
    new_width = int(math.ceil(W / float(S)))
    new_height = int(math.ceil(H / float(S)))
    pad_needed_height = (new_height - 1)  * S + F - W
    pad_top = int(pad_needed_height / 2.)
    pad_down = pad_needed_height - pad_top
    pad_needed_width = (new_width - 1)  * S + F - W
    pad_left = int(pad_needed_width / 2.)
    pad_right = pad_needed_width - pad_left
    if tf:
        return ((pad_top, pad_left, pad_right, pad_down), (new_height, new_width))
    else:
        return ((1,1,1,1), (new_height, new_width))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s','--stage',
        type=str,
        default='deploy',
        help='The stage of prototxt, train|test|deploy.'
    )
    parser.add_argument(
        '-d','--lmdb',
        type=str,
        default="",
        help='The training or testing database'
    )
    parser.add_argument(
        '-l','--label-map',
        type=str,
        default="labelmap.prototxt",
        help='The label map for ssd training.'
    )
    parser.add_argument(
        '--classifier',
        action='store_true',
        help='Default generate ssd, if this is set, generate classifier prototxt.'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=1.0,
        help='The size of mobilenet channels, support 1.0, 0.75, 0.5, 0.25.'
    )
    parser.add_argument(
        '-c', '--class-num',
        type=int,
        required=True,
        help='Output class number, include the \'backgroud\' class. e.g. 21 for voc.'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.001,
        help='eps parameter of BatchNorm layers, default is 1e-5'
    )
    parser.add_argument(
        '--relu6',
        action='store_true',
        help='replace ReLU layers by ReLU6'
    )
    parser.add_argument(
        '--tfpad',
        action='store_true',
        help='use tensorflow pad=SAME'
    )
    parser.add_argument(
        '--nobn',
        action='store_true',
        help='do not use batch_norm, defualt is false'
    )
    parser.add_argument(
        '-v','--version',
        type=str,
        default="2",
        help='MobileNet version, 1|2'
    )
    parser.add_argument(
        '-t','--type',
        type=str,
        default="ssdlite",
        help='ssd type, ssd|ssdlite'
    )
    FLAGS, unparsed = parser.parse_known_args()
    net_specs = caffe_pb2.NetParameter()
    net = CaffeNetGenerator(net_specs)
    net.generate(FLAGS)
    print text_format.MessageToString(net_specs, float_format=".5g")
