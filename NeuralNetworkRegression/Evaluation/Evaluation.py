import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

X = tf.range(0, 100)
y = tf.add(X, 32)
# plt.scatter(X, y)
# plt.show()

# Split Train and Test
X_train = X[:80]
X_test = X[80:]

y_train = y[:80]
y_test = y[80:]

# Visualise
plt.figure(figsize=(20, 14))
plt.scatter(X_train, y_train, c="b", label="Training Data")
plt.scatter(X_test, y_test, c="g", label="Testing Data")
plt.legend()
plt.show()

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, name="output_layer"),
], name="ModelOne")

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500)

model.summary()

plot_model(model=model, show_shapes=True)

y_pred = model.predict(X_test)

print(y_test, "\n")
print(tf.squeeze(y_pred), "\n")


def predict_function(training_data=X_train,
                     training_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     pred=y_pred):
    plt.figure(figsize=(20, 14))
    plt.scatter(training_data, training_labels, c="b", label="Training Data")
    plt.scatter(test_data, test_labels, c="g", label="Test Data")
    plt.scatter(test_data, pred, c="r", label="Prediction")
    plt.legend()
    plt.show()


predict_function()

# Evaluate the test data

model.evaluate(X_test, y_test)

# mae
print(tf.metrics.mean_absolute_error(y_test, tf.squeeze(y_pred)), "\n")

# mse
print(tf.metrics.mean_squared_error(y_test, tf.squeeze(y_pred)), "\n")

