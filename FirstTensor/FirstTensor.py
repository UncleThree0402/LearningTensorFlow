import tensorflow as tf

# Check Version
print(tf.__version__)

scalar = tf.constant(10)
print(scalar)
print(scalar.ndim)

vector = tf.constant([[10, 10]])
print(vector)
print(vector.ndim)

matrix = tf.constant([[10, 10],
                      [10, 10]])
print(matrix)
print(matrix.ndim)

other_matrix = tf.constant([[10., 10.],
                            [10., 10.]], dtype=tf.float16)
print(other_matrix)
print(other_matrix.ndim)

another_matrix = tf.constant([[10., 10.],
                              [10., 10.],
                              [10., 10.]], dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)

tensor = tf.constant(
    [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
print(tensor)
print(tensor.ndim)
