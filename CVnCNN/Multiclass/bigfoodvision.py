import tensor_tool.file
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tensor_tool.file.check_file("../../Dataset/10_food_classes_all_data")
tensor_tool.file.get_classes_name("../../Dataset/10_food_classes_all_data/train")

# Preprocess
tf.random.set_seed(42)

train_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                    rotation_range=0.3,
                                    shear_range=0.3,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

valid_data_gen = ImageDataGenerator(rescale=1. / 255.,
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
plt.show()
