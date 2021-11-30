import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.range(0, 100)
y = tf.add(X, 32)
plt.scatter(X, y)
# plt.show()

# Split Train and Test
X_train = X[:80]
X_test = X[80:]

y_train = y[:80]
y_test = y[80:]

# Visualise
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c="b", label="Training Data")
plt.scatter(X_test, y_test, c="g", label="Testing Data")
plt.legend()
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500)

model.summary()