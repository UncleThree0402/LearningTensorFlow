import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PlotGraph.confusion_matrix
from PlotGraph import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check File
for dirpath, dirnames, filenames in os.walk("../Dataset/pizza_steak"):
    print(f"There are {len(dirnames)}, {len(filenames)} files in {dirpath}")

steak_image_train = len(os.listdir("../Dataset/pizza_steak/train/steak"))
steak_image_test = len(os.listdir("../Dataset/pizza_steak/test/steak"))
pizze_image_train = len(os.listdir("../Dataset/pizza_steak/train/pizza"))
pizze_image_test = len(os.listdir("../Dataset/pizza_steak/test/pizza"))

print(
    f"Steak have {steak_image_train} train images and {steak_image_test} test images, Pizza have {pizze_image_train} train images and {pizze_image_test} test images")

# Get classes names
data_dir = pathlib.Path("../Dataset/pizza_steak/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

# Get random image
img = image.view_random_image("../Dataset/pizza_steak/train/", "steak")

# Create a tensor
print(tf.constant(img))

# Normalize
print(img / 255.0)

# Preprocess by tensorflow
tf.random.set_seed(42)

train_data_gen = ImageDataGenerator(rescale=1. / 255)
valid_data_gen = ImageDataGenerator(rescale=1. / 255)

## Path
train_dir = "../Dataset/pizza_steak/train/"
test_dir = "../Dataset/pizza_steak/test/"

## Create batches
train_data = train_data_gen.flow_from_directory(directory=train_dir,
                                                batch_size=32,
                                                target_size=(224, 224),
                                                class_mode="binary",
                                                seed=42)

valid_data = valid_data_gen.flow_from_directory(directory=test_dir,
                                                batch_size=32,
                                                target_size=(224, 224),
                                                class_mode="binary",
                                                seed=42)

# Tiny VGG

model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_data, epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

pd.DataFrame(history_1.history).plot()
plt.title("CNN")
plt.show()

# Non-cnn model
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_2 = model_2.fit(train_data, epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

pd.DataFrame(history_2.history).plot()
plt.title("Non-CNN")
plt.show()

# Non-cnn model with 25x more node inside Dense layer
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_3 = model_3.fit(train_data, epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

pd.DataFrame(history_3.history).plot()
plt.title("Non-CNN with 25x more node inside Dense layers")
plt.show()

model_1.summary()
model_2.summary()
model_3.summary()
