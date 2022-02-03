from tensor_tool import image_data, file, callbacks
from plot_graph import loss_accuracy, image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Check File
file.check_file("../Dataset/10_food_classes_10_percent")
file.check_file("../Dataset/10_food_classes_1_percent")
file.check_file("../Dataset/10_food_classes_all_data")

# Preprocess
train_data_10, test_data_10 = image_data.get_dataset("../Dataset/10_food_classes_10_percent/train",
                                                     "../Dataset/10_food_classes_10_percent/test",
                                                     class_mode="categorical")

train_data_1, test_data_1 = image_data.get_dataset("../Dataset/10_food_classes_1_percent/train",
                                                   "../Dataset/10_food_classes_1_percent/test",
                                                   class_mode="categorical")

train_data_all, test_data_all = image_data.get_dataset("../Dataset/10_food_classes_all_data/train",
                                                       "../Dataset/10_food_classes_all_data/test",
                                                       class_mode="categorical")

# Class Names
# print(train_data.class_names)

# How Data Look like
# for images, labels in train_data.take(1):
#     print(images, labels)


tf.random.set_seed(42)

# Base model using tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

base_model.trainable = False

# Input layer
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# Model_0
x = base_model(inputs)
print(f"base_model shape {x.shape}")

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
print(f"GlobalAveragePooling2D shape {x.shape}")

outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_0 = model_0.fit(train_data_10,
                        epochs=5,
                        steps_per_epoch=len(train_data_10),
                        validation_data=test_data_10,
                        validation_steps=int(0.25 * len(test_data_10)),
                        callbacks=[
                            callbacks.create_tensorboard_callback("keras_application", "EfficientNetB0_baseline")])

# Evaluate
model_0_result = model_0.evaluate(test_data_10)

# Application Layer
# base_model.summary()

loss_accuracy.plot_loss_curves(history_0, "model_0 ")
loss_accuracy.plot_accuracy_curves(history_0, "model_0 ")

# model_1
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = image_data.augmentation_layer()(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAvgPool2D(name="global_avh_pool")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model_1 = tf.keras.Model(inputs, outputs)

model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_data_1,
                        epochs=5,
                        steps_per_epoch=len(train_data_1),
                        validation_data=test_data_1,
                        validation_steps=int(0.25 * len(test_data_1)),
                        callbacks=[callbacks.create_tensorboard_callback("keras_application", "EfficientNetB0_model1"),
                                   callbacks.create_model_checkpoint("model_1_checkpoint")])

model_1_result = model_1.evaluate(test_data_1)

loss_accuracy.plot_loss_curves(history_1, "model_1 ")
loss_accuracy.plot_accuracy_curves(history_1, "model_1 ")

# model_2
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = image_data.augmentation_layer()(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAvgPool2D(name="global_avh_pool")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_2 = model_2.fit(train_data_10,
                        epochs=5,
                        steps_per_epoch=len(train_data_10),
                        validation_data=test_data_10,
                        validation_steps=int(0.25 * len(test_data_10)),
                        callbacks=[callbacks.create_tensorboard_callback("keras_application", "EfficientNetB0_model2"),
                                   callbacks.create_model_checkpoint("model_2_checkpoint")])

model_2_og_result = model_2.evaluate(test_data_10)
model_2_result = model_2_og_result
model_2.load_weights("model_2_checkpoint/checkpoint.ckpt")
model_2_loaded_result = model_2.evaluate(test_data_10)

print(model_2_og_result, model_2_loaded_result)

loss_accuracy.plot_loss_curves(history_2, "model_2 ")
loss_accuracy.plot_accuracy_curves(history_2, "model_2 ")

for layer in model_2.layers:
    print(layer, layer.trainable)

for i, layer in enumerate(model_2.layers[2].layers):
    print(i, layer.name, layer.trainable)

print(len(model_2.layers[2].trainable_variables))

# model_3 (tune model_2)
base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # TUNE
                metrics=["accuracy"])

for i, layer in enumerate(model_2.layers[2].layers):
    print(i, layer.name, layer.trainable)

print(len(model_2.layers[2].trainable_variables))

history_3 = model_2.fit(train_data_10,
                        epochs=10,
                        steps_per_epoch=len(train_data_10),
                        validation_data=test_data_10,
                        validation_steps=int(0.25 * len(test_data_10)),
                        initial_epoch=history_2.epoch[-1],  # Last epoch
                        callbacks=[callbacks.create_tensorboard_callback("keras_application", "EfficientNetB0_model3"),
                                   callbacks.create_model_checkpoint("model_3_checkpoint")])

model_3_result = model_2.evaluate(test_data_10)

loss_accuracy.plot_loss_curves(history_3, "model_3 ")
loss_accuracy.plot_accuracy_curves(history_3, "model_3 ")
# model_4
model_2.load_weights("model_2_checkpoint/checkpoint.ckpt")

base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # TUNE
                metrics=["accuracy"])

history_4 = model_2.fit(train_data_all,
                        epochs=10,
                        steps_per_epoch=len(train_data_all),
                        validation_data=test_data_10,
                        validation_steps=int(0.25 * len(test_data_all)),
                        initial_epoch=history_2.epoch[-1],  # Last epoch
                        callbacks=[callbacks.create_tensorboard_callback("keras_application", "EfficientNetB0_model4"),
                                   callbacks.create_model_checkpoint("model_4_checkpoint")])

model_4_result = model_2.evaluate(test_data_all)

loss_accuracy.plot_loss_curves(history_4, "model_4 ")
loss_accuracy.plot_accuracy_curves(history_4, "model_4 ")

print(f"\n Module 1 : {model_1_result}")
print(f"\n Module 2 : {model_2_result}")
print(f"\n Module 3 : {model_3_result}")
print(f"\n Module 4 : {model_4_result}")