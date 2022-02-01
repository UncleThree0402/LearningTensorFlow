import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
import random

import PlotGraph as pg
import PlotGraph.confusion_matrix

(train_data, train_labels) , (test_data, test_labels) = fashion_mnist.load_data()

# Check how the data look like
# print(f"Train Data : \n {train_data[0]} \n")
# print(f"Train Label : \n {train_labels[0]} \n")

# Shape of Data
# print(train_data[0].shape, train_labels[0].shape)

# Show image
# plt.imshow(train_data[7], cmap=plt.cm.gray)
# plt.show()

# Label the classes
image_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                 "Ankle Boot"]

# Test
# index_image = 15
# plt.imshow(train_data[index_image], cmap=plt.cm.binary)
# plt.show()
# print(image_classes[train_labels[index_image]])


# Plot random image
plt.figure(figsize=(4,4))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    random_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[random_index], cmap=plt.cm.binary)
    plt.title(image_classes[train_labels[random_index]])
    plt.axis(False)
plt.show()

tf.random.set_seed(42)
# Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax"),
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(train_data, tf.one_hot(train_labels, depth=10),epochs=10,
          validation_data=(test_data, tf.one_hot(test_labels, depth=10)))

# Find min & max
print(train_data.min(), train_data.max())

# Normalize
train_data_nor = train_data / 255.0
test_data_nor = test_data / 255.0

# Min & Max after normalize
print(train_data_nor.min(), train_data_nor.max())

# Normalized Model

model_norm = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model_norm.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])

history_norm = model_norm.fit(train_data_nor, tf.one_hot(train_labels, depth=10), epochs=10,
                              validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)))

# Find learning rate

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 20))

model_lr = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model_lr.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])

history_lr = model_lr.fit(train_data_nor, tf.one_hot(train_labels, depth=10), epochs=100,
                              validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)),
                          callbacks=[lr_scheduler])

lrs = 1e-4 * 10 ** (tf.range(100) / 20)

plt.figure(figsize=(10,7))
plt.semilogx(lrs, history_lr.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
plt.show()

# Model after tweaking learning rate

model_tweak = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_tweak.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=["accuracy"])

history_tweak = model_tweak.fit(train_data_nor, tf.one_hot(train_labels, depth=10),
                                epochs=20,
                                validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)))

plt.figure(figsize=(10,10))
ax1 = plt.subplot(2,2,1)
ax1.plot(pd.DataFrame(history.history))
ax1.set_title("Non-normalize data")
ax2 = plt.subplot(2,2,2)
ax2.plot(pd.DataFrame(history_norm.history))
ax2.set_title("Normalize data")
ax3 = plt.subplot(2,2,3)
ax3.plot(pd.DataFrame(history_tweak.history))
ax3.set_title("Tweaked Data")
plt.show()

# Predict

y_prob = model_tweak.predict(test_data_nor)
y_pred = y_prob.argmax(axis=1)
PlotGraph.confusion_matrix.plot_confusion_matrix(y_true=test_labels, y_preds=y_pred, classes=image_classes, figsize=(15,15),text_size=10)

