import tensorflow as tf

some_list = [0, 1, 2, 3, 4, 5]

# One Hot Encoding
print(tf.one_hot(some_list, depth=6), "\n")

# With OnValue and OffValue
print(tf.one_hot(some_list, depth=6, on_value=True, off_value=False))
