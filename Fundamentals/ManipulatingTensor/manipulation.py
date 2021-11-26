import tensorflow as tf

tensorA = tf.ones(shape=(4, 4))

print(tensorA)

# Addition
print(tensorA + 10)

# Subtraction
print(tensorA - 10)

# Multiplication
print(tensorA * 10)

# using tensorflow method

print(tf.multiply(tensorA, 1.5))

print(tf.divide(tensorA, 0.5))
