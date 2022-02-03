import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def get_classes_name(train_dir):
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    print(class_names)
    return class_names


# CPU
def get_data(train_dir, test_dir, valid_dir=None, class_mode=None, augmented=True):
    if augmented:
        train_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                            rotation_range=0.3,
                                            shear_range=0.3,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)

        test_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                           rotation_range=0.3,
                                           shear_range=0.3,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        if valid_dir:
            valid_data_gen = ImageDataGenerator(rescale=1. / 255.,
                                                rotation_range=0.3,
                                                shear_range=0.3,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)
    else:
        train_data_gen = ImageDataGenerator(rescale=1. / 255.)

        test_data_gen = ImageDataGenerator(rescale=1. / 255.)

        if valid_dir:
            valid_data_gen = ImageDataGenerator(rescale=1. / 255.)

    train_data = train_data_gen.flow_from_directory(train_dir,
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode=class_mode,
                                                    seed=SEED)

    test_data = test_data_gen.flow_from_directory(test_dir,
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode=class_mode,
                                                  seed=SEED)

    if valid_dir:

        valid_data = valid_data_gen.flow_from_directory(valid_dir,
                                                        target_size=IMG_SIZE,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=class_mode,
                                                        seed=SEED)

        return train_data, test_data, valid_data
    else:
        return train_data, test_data


# GPU
def get_dataset(train_dir, test_dir, valid_dir=None, class_mode=None):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                     image_size=IMG_SIZE,
                                                                     label_mode=class_mode,
                                                                     batch_size=BATCH_SIZE)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                    image_size=IMG_SIZE,
                                                                    label_mode=class_mode,
                                                                    batch_size=BATCH_SIZE,
                                                                    shuffle=False)

    if valid_dir:
        valid_data = tf.keras.preprocessing.image_dataset_from_directory(directory=valid_dir,
                                                                         image_size=IMG_SIZE,
                                                                         label_mode=class_mode,
                                                                         batch_size=BATCH_SIZE, )

        return train_data, valid_data, test_data
    else:
        return train_data, test_data


# GPU
def augmentation_layer():
    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
    ], name="data_augmentation")
    return data_augmentation


def load_and_pred_image(filename, img_shape=224, scale=True):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    if scale:
        img = img / 255.
    return img


def predict_with_model(filename, model, classes_name, scale=True):
    img = load_and_pred_image(filename, scale=scale)
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    pred_probs = model.predict(img)
    if len(classes_name) <= 2:
        pred_class = classes_name[int(tf.round(pred_probs))]
    else:
        pred_class = classes_name[int(tf.argmax(pred_probs, axis=1))]
    img = load_and_pred_image(filename)
    plt.imshow(img)
    plt.axis(False)
    plt.title(f"Prediction : {pred_class} ({pred_probs.max():.2f})")
    plt.show()


def predict_random_data(model, test_dir, class_names):
    plt.figure(figsize=(17, 10))
    for i in range(3):
        class_name = random.choice(class_names)
        file_name = random.choice(os.listdir(test_dir + "/" + class_name))
        file_path = test_dir + "/" + class_name + "/" + file_name
        img = load_and_pred_image(file_path, scale=False)
        pred_probs = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[int(tf.argmax(pred_probs, axis=1))]

        plt.subplot(1, 3, i + 1)
        plt.imshow(img / 255.)
        if class_name == pred_class:
            colour = "g"
        else:
            colour = "r"
        plt.title(f"Actual : {class_name}, Pred : {pred_class}, prob : {pred_probs.max():.2f}", c=colour)
        plt.axis(False)
    plt.show()


def create_y_labels_y_pred(pred_probs, test_data):
    pred_classes = tf.argmax(pred_probs, axis=1)

    y_labels = []

    for image, labels in test_data.unbatch():
        y_labels.append(labels.numpy().argmax())

    return y_labels, pred_classes
