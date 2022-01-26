from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import itertools

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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=42)

tf.random.seed=42

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_1")

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
              metrics=["accuracy"])

early = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=20)
history = model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[early])

losses , accuracy = model.evaluate(X_test, y_test)

print(f"Model loss : {losses}")
print(f"Model accuracy : {accuracy:%}")

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title(model.name + " Loss Curves")
plt.show()

plot_decision_boundary(model, X, y)

y_pred = model.predict(X_test)

print(y_pred[:10])

print(tf.round(y_pred)[:10])

# Confusion Matrix
cm = confusion_matrix(y_test, tf.round(y_pred))

# Get Normalize
cm_normalize = cm.astype("float") / cm.sum(axis=1) [: , np.newaxis]

print(cm)

# Classes
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize=(10,10))

cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)
classes = False

if classes:
    labels = classes
else:
    labels = np.arange(cm.shape[0])

ax.set(title="Confusion Matrix",
       xlabel="Predicted Table",
       ylabel="True Table",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

# Set Label
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
ax.title.set_size(20)

threshold = (cm.max() + cm.min()) / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_normalize[i , j] * 100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i , j] > threshold else "black",
             size=15)
plt.show()
