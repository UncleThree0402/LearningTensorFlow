import tensorflow as tf

random = tf.random.Generator.from_seed(69)

random = random.normal(shape=(3, 3))

print(random)


# tf.random.set_seed(42)
# without global seed set seed will set to be operation seed
random = tf.random.shuffle(random)

print(random)
