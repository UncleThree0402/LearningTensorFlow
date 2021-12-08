import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)
    if len(y_pred[0]) > 1:
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        y_pred = np.round(y_pred).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.colorbar()
    plt.title(model.name + " Decision Boundary")
    plt.show()


# Create Circle
X, y = make_circles(2000,
                    noise=0.03,
                    random_state=42)

circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

# plt.figure(figsize=(15, 12))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Shape : X = (2000, 2) y = (2000,)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=42)

# Create Model

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
], name="model_1")

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
], name="model_2")

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
], name="model_3")

# linear activation
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="linear")
], name="model_4")

# non-linear activation
model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu")
], name="model_5")

# more layers
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_6")

model_6.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"])

history = model_6.fit(train_X, train_y, epochs=100, verbose=1)

plot_model(model=model_6, to_file="Image/model_6.png", show_shapes=True)

print(model_6.evaluate(test_X, test_y))

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title(model_6.name + " Loss Curves")
plt.show()

plot_decision_boundary(model_6, X, y)
