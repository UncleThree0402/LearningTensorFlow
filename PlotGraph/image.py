import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
import os


def view_random_image(target_dir, class_name):
    folder = target_dir + class_name
    random_image = random.sample(os.listdir(folder), 1)
    print(random_image)
    img = mpimg.imread(folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(class_name)
    plt.axis("off")
    plt.show()
    print(img.shape)
    return img


def load_and_pred_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img
