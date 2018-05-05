#!/bin/sh
mkdir -p snapshot
../../build/tools/caffe train -solver="solver_train.prototxt" \
-weights="deploy_voc.caffemodel" \
-gpu 0
