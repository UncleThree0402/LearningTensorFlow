import tensorflow as tf

tensorA = tf.ones(shape=(4, 4))

print(tensorA, "\n")

# Addition
print(tensorA + 10, "\n")

# Subtraction
print(tensorA - 10, "\n")

# Multiplication
print(tensorA * 10, "\n")

# using tensorflow method

print(tf.multiply(tensorA, 1.5), "\n")

print(tf.divide(tensorA, 0.5), "\n")

# square
tensorA = tf.range(1, 10)
print(tensorA, "\n")

print(tf.math.square(tensorA), "\n")

# sqrt
print(tf.math.sqrt(tf.cast(tensorA, dtype=tf.float32)), "\n")

# log
print(tf.math.log(tf.cast(tensorA, dtype=tf.float32)), "\n")

# matrix manipulation

randomOne = tf.random.Generator.from_seed(100)
randomOne = randomOne.uniform(shape=(2, 3))

print(randomOne, "\n")

randomTwo = tf.random.Generator.from_seed(50)
randomTwo = randomTwo.uniform(shape=(3, 2))
print(randomTwo, "\n")

print(randomOne @ randomTwo, "\n")
print(tf.matmul(randomOne, randomTwo), "\n")
print(tf.tensordot(randomOne, randomTwo, axes=0), "\n")
# reshape

print(randomOne @ tf.reshape(randomOne, shape=(3, 2)), "\n")

# transpose

print(tf.transpose(randomOne), "\n")
