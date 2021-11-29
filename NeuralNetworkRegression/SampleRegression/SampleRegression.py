import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-10, 20, dtype=np.float32)
y = np.arange(10, 40, dtype=np.float32)
plt.scatter(X, y)
plt.show()

X = tf.constant(X)
y = tf.constant(y)

# Set seed
tf.random.set_seed(42)

# Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.fit(tf.expand_dims(X, axis=-1), y, epochs=5000)

print(model.predict([10]))
