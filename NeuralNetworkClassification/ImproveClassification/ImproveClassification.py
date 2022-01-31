import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.utils import plot_model
import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


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


X, y = make_circles(2000, noise=0.03, random_state=42)

circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=42)

tf.random.set_seed = 42

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="Model")

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
              metrics=["accuracy"])
# Metrics for classification : accuracy, precision(less fp), recall(less fn), f1_scores, confusion

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 20))
early = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=5)
lrs = 1e-4 * 10 ** (tf.range(100) / 20)
history = model.fit(train_X, train_y, epochs=100, verbose=1)
# history = model.fit(train_X, train_y, epochs=100, verbose=1, callbacks=[lr_scheduler])

losses , accuracy = model.evaluate(test_X, test_y)

print(f"Model loss : {losses}")
print(f"Model accuracy : {accuracy:%}")

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title(model.name + " Accuracy Curves")
plt.show()

# plt.figure(figsize=(10,7))
# plt.semilogx(lrs, history.history["loss"])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss")
# plt.title("Learning Rate vs Loss")
# plt.show()

plot_decision_boundary(model, X, y)
