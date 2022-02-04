import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd
from tensor_tool import image_data, callbacks
import matplotlib.pyplot as plt

dataset_list = tfds.list_builders()
print("food101" in dataset_list)

(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             data_dir="../Dataset")

print(ds_info.features)

# Classes
class_names = ds_info.features["label"].names

print(class_names[:10])

# Take One
take_one_sample = train_data.take(1)
print(take_one_sample)

# Plot
for image, label in take_one_sample:
    print(f"Image Shape {image.shape}, Image dtype {image.dtype}, class {label}, name {class_names[label.numpy()]}")
    plt.imshow(image)
    plt.title(class_names[label.numpy()])
    plt.axis(False)
    plt.show()

"""
Datatype : uint8
Value : 0 - 255 
"""

preprocess_image = image_data.preprocess_image(image, label)[0]
print(
    f"new Image Shape {preprocess_image.shape}, new Image dtype {preprocess_image.dtype}, new class {label}, new name {class_names[label.numpy()]}")

# batch & ready
train_data = train_data.map(map_func=image_data.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(map_func=image_data.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

test_data = test_data.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)

print(train_data, test_data)

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
augmentation_layer = image_data.augmentation_layer()(inputs)
x = base_model(augmentation_layer, training=False)
x = tf.keras.layers.GlobalAvgPool2D(name="gap2d")(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_0.fit(train_data,
            epochs=5,
            steps_per_epoch=len(train_data),
            validation_data=test_data,
            validation_steps=len(test_data),
            callbacks=[callbacks.create_tensorboard_callback("food101_model", "baseline"),
                       callbacks.create_model_checkpoint("baseline_checkpoint")])
