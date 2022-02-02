import tensor_tool.file
import tensor_tool.image_data
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tensor_tool.file.check_file("../../Dataset/10_food_classes_all_data")
tensor_tool.image_data.get_classes_name("../../Dataset/10_food_classes_all_data/train")

# Preprocess
tf.random.set_seed(42)

train_data_gen = ImageDataGenerator(rescale=1. / 255.)

valid_data_gen = ImageDataGenerator(rescale=1. / 255.)

train_data_gen_augmented = ImageDataGenerator(rescale=1. / 255.,
                                              rotation_range=0.3,
                                              shear_range=0.3,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)

train_dir = "../../Dataset/10_food_classes_all_data/train"
test_dir = "../../Dataset/10_food_classes_all_data/test"

train_date = train_data_gen.flow_from_directory(train_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode="categorical",
                                                seed=42)

valid_data = valid_data_gen.flow_from_directory(test_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode="categorical",
                                                seed=42)

train_date_augmented = train_data_gen_augmented.flow_from_directory(train_dir,
                                                                    target_size=(224, 224),
                                                                    batch_size=32,
                                                                    class_mode="categorical",
                                                                    seed=42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_date,
                        epochs=5,
                        steps_per_epoch=len(train_date),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

print(f"\n {model_1.evaluate(valid_data)}")

pd.DataFrame(history_1.history).plot()
plt.title("Tiny VGG (Not augmented)")
plt.show()

model_1.save("tiny_vgg_not_augmented")

model_2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_2 = model_2.fit(train_date_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_date_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

print(f"\n {model_2.evaluate(valid_data)}")

pd.DataFrame(history_2.history).plot()
plt.title("Tiny VGG (augmented)")
plt.show()

model_2.save("tiny_vgg_augmented")

model_3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_3.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_3 = model_3.fit(train_date_augmented,
                        epochs=10,
                        steps_per_epoch=len(train_date_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

print(f"\n {model_3.evaluate(valid_data)}")

pd.DataFrame(history_3.history).plot()
plt.title("Tiny VGG (augmented 10 epochs)")
plt.show()

model_3.save("tiny_vgg_augmented_10_epochs")

model_4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(5, 5, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 5, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(15, 5, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(20, 5, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_4.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_4 = model_4.fit(train_date_augmented,
                        epochs=10,
                        steps_per_epoch=len(train_date_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

print(f"\n {model_4.evaluate(valid_data)}")

pd.DataFrame(history_4.history).plot()
plt.title("Self CNN")
plt.show()

model_4.save("self_CNN")

model_1.summary()
model_2.summary()
model_3.summary()
model_4.summary()

