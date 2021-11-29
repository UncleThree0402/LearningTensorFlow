import tensorflow as tf

randomOne = tf.random.Generator.from_seed(42)
randomOne = randomOne.uniform(shape=(4, 4), dtype=tf.float32)

print(tf.cast(randomOne, dtype=tf.float16))
