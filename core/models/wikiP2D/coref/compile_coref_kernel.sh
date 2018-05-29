#!/bin/bash
TENSORFLOW=$(python -c 'import tensorflow as tf; print(tf.__path__[0])')
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_INC2=$TF_INC/external/nsync/public
# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
