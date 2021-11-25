import tensorflow as tf
import numpy as np

tensorA = tf.ones(shape=(2, 3, 4, 5), dtype=tf.int32)
print(tensorA)

print("Rank of tensor : ", tensorA.ndim)
print("Shape of tensor : ", tensorA.shape)
print("Type of tensor : ", tensorA.dtype)
print("Shape of first axis", tensorA.shape[0])
print("Shape of last axis", tensorA.shape[-1])
print("Size of tensor : ", tf.size(tensorA))
print("Size of tensor : ", tf.size(tensorA).numpy())

