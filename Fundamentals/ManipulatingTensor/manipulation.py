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

# matrix manipulation

randomOne = tf.random.Generator.from_seed(100)
randomOne = randomOne.uniform(shape=(2, 3))

print(randomOne)

randomTwo = tf.random.Generator.from_seed(50)
randomTwo = randomTwo.uniform(shape=(3, 2))
print(randomTwo)

print(randomOne @ randomTwo)
print(tf.matmul(randomOne, randomTwo))
print(tf.tensordot(randomOne, randomTwo))
# reshape

print(randomOne @ tf.reshape(randomOne, shape=(3, 2)))

# transpose

print(tf.transpose(randomOne))



