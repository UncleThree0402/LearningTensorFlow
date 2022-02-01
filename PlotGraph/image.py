import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
