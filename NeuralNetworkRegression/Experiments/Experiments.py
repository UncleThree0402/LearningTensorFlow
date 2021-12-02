import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

X = tf.range(-10, 10, dtype=tf.int32)
y = tf.subtract(X, 24)

X_train = X[:16]
y_train = y[:16]

X_test = X[16:]
y_test = y[16:]

tf.random.set_seed(42)


def plot_pred(training_data=X_train,
              training_label=y_train,
              testing_data=X_test,
              testing_label=y_test,
              *prediction):
    plt.figure(figsize=(10, 7))
    plt.scatter(training_data, training_label, c="b", label="Training")
    plt.scatter(testing_data, testing_label, c="g", label="Testing")
    plt.scatter(testing_data, prediction, c="r", label="Prediction")
    plt.legend()
    plt.show()


# First Model
modelOne = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
], name="ModelOne")

modelOne.compile(loss=tf.keras.losses.mae,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=["mae"])

modelOne.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

modelOnePred = tf.squeeze(modelOne.predict(X_test))

plot_pred(X_train, y_train, X_test, y_test, modelOnePred)

print("\n")

# Second Model
modelTwo = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1)
], name="ModelTwo")

modelTwo.compile(loss=tf.keras.losses.mae,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=["mae"])

modelTwo.fit(X_train, y_train, epochs=100)

modelTwoPred = tf.squeeze(modelTwo.predict(X_test))

plot_pred(X_train, y_train, X_test, y_test, modelTwoPred)

print("\n")

# Third Model
modelThree = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1)
], name="ModelThree")

modelThree.compile(loss=tf.keras.losses.mae,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   metrics=["mae"])

modelThree.fit(X_train, y_train, epochs=500)

modelThreePred = tf.squeeze(modelThree.predict(X_test))

plot_pred(X_train, y_train, X_test, y_test, modelThreePred)

print("\n")

print("mae : ", tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelOnePred)).numpy())
print("mse : ", tf.metrics.mean_squared_error(y_test, tf.squeeze(modelOnePred)).numpy(), "\n")

print("mae : ", tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelTwoPred)).numpy())
print("mse : ", tf.metrics.mean_squared_error(y_test, tf.squeeze(modelTwoPred)).numpy(), "\n")

print("mae : ", tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelThreePred)).numpy())
print("mse : ", tf.metrics.mean_squared_error(y_test, tf.squeeze(modelThreePred)).numpy(), "\n")
