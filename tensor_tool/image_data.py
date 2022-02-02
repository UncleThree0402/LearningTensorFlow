import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def get_classes_name(train_dir):
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    print(class_names)
    return class_names


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


def load_and_pred_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img
