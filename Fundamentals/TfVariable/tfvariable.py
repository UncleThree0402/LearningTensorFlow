import tensorflow as tf

# Change declare with tr.Variable()
changeableVector = tf.Variable([1, 2])
print(changeableVector)

# Change with  .assign
changeableVector[1].assign(10)
print(changeableVector)

changeableMatrix = tf.Variable([[1, 2], [3, 4]])
print(changeableMatrix)

changeableMatrix[0, 1].assign(12)
print(changeableMatrix)
