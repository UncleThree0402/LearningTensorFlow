import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import pandas as pd
import os

# For window
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

X = tf.range(-10, 10, dtype=tf.int32)
y = tf.subtract(X, 24)

X_train = X[:16]
y_train = y[:16]

X_test = X[16:]
y_test = y[16:]

tf.random.set_seed(42)


def plot_pred(prediction,
              title,
              training_data=X_train,
              training_label=y_train,
              testing_data=X_test,
              testing_label=y_test,):
    plt.figure(figsize=(10, 7))
    plt.scatter(training_data, training_label, c="b", label="Training")
    plt.scatter(testing_data, testing_label, c="g", label="Testing")
    plt.scatter(testing_data, prediction, c="r", label="Prediction")
    plt.legend()
    plt.title(title)
    plt.show()


# First Model
modelOne = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
], name="ModelOne")

modelOne.compile(loss=tf.keras.losses.mae,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=["mae"])

modelOne.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

modelOnePred = tf.squeeze(modelOne.predict(X_test))

plot_pred(prediction=tf.squeeze(modelOnePred), title=modelOne.name)

plot_model(model=modelOne, to_file="model_one.png", show_shapes=True)

print("\n")

# Second Model
modelTwo = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
], name="ModelTwo")

modelTwo.compile(loss=tf.keras.losses.mae,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=["mae"])

modelTwo.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

modelTwoPred = tf.squeeze(modelTwo.predict(X_test))

plot_pred(prediction=modelTwoPred, title=modelTwo.name)

plot_model(model=modelTwo, to_file="model_two.png", show_shapes=True)

print("\n")

# Third Model
modelThree = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
], name="ModelThree")

modelThree.compile(loss=tf.keras.losses.mae,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   metrics=["mae"])

modelThree.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500)

modelThreePred = tf.squeeze(modelThree.predict(X_test))

plot_pred(prediction=modelThreePred, title=modelThree.name)

plot_model(model=modelThree, to_file="model_three.png", show_shapes=True)

print("\n")

modelOneMae = tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelOnePred)).numpy()
modelOneMse = tf.metrics.mean_squared_error(y_test, tf.squeeze(modelOnePred)).numpy()

modelTwoMae = tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelTwoPred)).numpy()
modelTwoMse = tf.metrics.mean_squared_error(y_test, tf.squeeze(modelTwoPred)).numpy()

modelThreeMae = tf.metrics.mean_absolute_error(y_test, tf.squeeze(modelThreePred)).numpy()
modelThreeMse = tf.metrics.mean_squared_error(y_test, tf.squeeze(modelThreePred)).numpy()

model_result = [["Model One", modelOneMae, modelOneMse],
                ["Model Two", modelTwoMae, modelTwoMse],
                ["Model Three", modelThreeMae, modelThreeMse]]

result = pd.DataFrame(model_result, columns=["Model", "mae", "mse"])
print(result, "\n")

modelOne.summary()
modelTwo.summary()
modelThree.summary()
