import tensorflow as tf

import numpy as np

# absolute value
tensorA = tf.constant([-6, - 4])
print(tf.abs(tensorA), "\n")

# random tensor
tensorB = tf.constant(np.random.randint(0, 100, 50, dtype=np.int32))
print(tensorB, "\n")

# Max
print(tf.reduce_max(tensorB))
print(np.max(tensorB), "\n")

# Min
print(tf.reduce_min(tensorB))
print(np.min(tensorB), "\n")

# Mean
print(tf.reduce_mean(tensorB))
print(np.mean(tensorB), "\n")

# Sum
print(tf.reduce_sum(tensorB))
print(np.sum(tensorB), "\n")

# Variance
print(tf.math.reduce_variance(tf.cast(tensorB, dtype=tf.float32)))
print(np.var(tensorB, dtype=np.float32), "\n")

# Standard deviation
print(tf.math.reduce_std(tf.cast(tensorB, dtype=tf.float32)))
print(np.std(tensorB, dtype=np.float32), "\n")

# Find positional min and max

randomOne = tf.random.Generator.from_seed(50)
randomOne = randomOne.uniform(shape=(50, ))
print(randomOne, "\n")

# Max position
print(tf.argmax(randomOne))
print(np.argmax(randomOne))
print(randomOne[np.argmax(randomOne)], "\n")

# Min position
print(tf.argmin(randomOne))
print(np.argmin(randomOne))
print(randomOne[np.argmin(randomOne)], "\n")

