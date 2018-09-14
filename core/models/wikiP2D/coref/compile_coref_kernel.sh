#!/bin/bash
# TENSORFLOW=$(python -c 'import tensorflow as tf; print(tf.__path__[0])')
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_INC2=$TF_INC/external/nsync/public
# # Linux (pip)
# g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0


# Build custom kernels.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
