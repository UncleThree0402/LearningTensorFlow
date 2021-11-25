import tensorflow as tf
import numpy as np

# As .ones at numpy
tfOne = tf.ones([3, 3])
print(tfOne)

tfZero = tf.zeros(shape=(3, 3))
print(tfZero)

# By numPy
numpyA = np.arange(1, 25, dtype=np.int32)
print(numpyA)

tensorA = tf.constant(numpyA)
print(tensorA)
print(tensorA.ndim)

tensorB = tf.constant(numpyA, shape=(2, 3, 4))
print(tensorB)
print(tensorB.ndim)
