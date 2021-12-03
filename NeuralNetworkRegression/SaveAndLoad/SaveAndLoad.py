import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.range(-10, 10, dtype=tf.int32)
y = tf.add(X, 12)

X_train = X[:16]
y_train = y[:16]

X_test = X[16:]
y_test = y[16:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(1)
], name="model_one")

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae", "mse"])

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500)

model.save(model.name)
model.save(model.name + ".h5")

model.summary()

modelTwo = tf.keras.models.load_model("model_one")

modelTwo.summary()

modelTwo = tf.keras.models.load_model("model_one.h5")

modelTwo.summary()