#!/bin/sh
mkdir -p snapshot
../../build/tools/caffe train -solver="solver_train.prototxt" \
-weights="deploy.caffemodel" \
-gpu 0
