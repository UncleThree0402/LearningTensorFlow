import tensorflow as tf
import numpy as np

tensorA = tf.ones(shape=(2, 3, 4, 5), dtype=tf.int32)
print(tensorA, "\n")

print("Rank of tensor : ", tensorA.ndim)
print("Shape of tensor : ", tensorA.shape)
print("Type of tensor : ", tensorA.dtype)
print("Shape of first axis", tensorA.shape[0])
print("Shape of last axis", tensorA.shape[-1])
print("Size of tensor : ", tf.size(tensorA))
print("Size of tensor : ", tf.size(tensorA).numpy())

# Find 0 to 2 item of each axis in tensor
print(tensorA[:2, :2, :2, :2], "\n")

# Find 0 to 2 item of each axis in tensor except last axis
print(tensorA[:2, :2, :2, :], "\n")

# Get last items of each row
print(tensorA[:, :, :, -1], "\n")

# New axis
tensorB = tensorA[..., tf.newaxis]
print(tensorB, "\n")

# New axis with other method

tensorC = tf.expand_dims(tensorA, axis=-1)
print(tensorC, "\n")

# Squeezing Tensor
tensorD = tf.random.Generator.from_seed(42)
tensorD = tensorD.uniform(shape=(1, 1, 1, 50))
print(tensorD, "\n")

print(tf.squeeze(tensorD))
