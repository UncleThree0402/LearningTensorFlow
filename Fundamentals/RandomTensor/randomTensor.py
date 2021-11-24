import tensorflow as tf

randomOne = tf.random.Generator.from_seed(42)
randomOne = randomOne.normal(shape=(3, 2))
print(randomOne)

randomOne = tf.random.Generator.from_seed(42)
randomOne = randomOne.uniform(shape=(3, 2))
print(randomOne)