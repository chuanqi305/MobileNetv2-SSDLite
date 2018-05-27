#!/bin/sh
../../build/tools/caffe train -solver="solver_test.prototxt" \
--weights=deploy.caffemodel \
-gpu 0
