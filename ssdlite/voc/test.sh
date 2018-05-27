#!/bin/sh
../../build/tools/caffe train -solver="solver_test.prototxt" \
--weights=deploy_voc.caffemodel \
-gpu 0
