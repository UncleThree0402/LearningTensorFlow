# Small Food Vision

## Dataset

steak & pizza from [Food101](https://www.kaggle.com/kmader/food41)

Steak:

![Steak](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/steak.png)

Pizza:

![Pizza](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/pizza.png)


## Check import file

```python
for dirpath, dirnames, filenames in os.walk("../Dataset/pizza_steak"):
    print(f"There are {len(dirnames)}, {len(filenames)} files in {dirpath}")

steak_image_train = len(os.listdir("../Dataset/pizza_steak/train/steak"))
steak_image_test = len(os.listdir("../Dataset/pizza_steak/test/steak"))
pizze_image_train = len(os.listdir("../Dataset/pizza_steak/train/pizza"))
pizze_image_test = len(os.listdir("../Dataset/pizza_steak/test/pizza"))

print(
    f"Steak have {steak_image_train} train images and {steak_image_test} test images, Pizza have {pizze_image_train} train images and {pizze_image_test} test images")
```

## Setup classes

```python
data_dir = pathlib.Path("../Dataset/pizza_steak/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)
```

## Preprocess with tensorflow

### Setup ImageDataGenerator

```python
train_data_gen = ImageDataGenerator(rescale=1. / 255)
valid_data_gen = ImageDataGenerator(rescale=1. / 255)
```

### Augmented Data
```python
train_data_gen_augmented = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=0.3,
                                              shear_range=0.3,
                                              zoom_range=0.3,
                                              width_shift_range=0.3,
                                              height_shift_range=0.3,
                                              horizontal_flip=True)
```

### Create batches

```python
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

train_data_augmented = train_data_gen_augmented.flow_from_directory(train_dir,
                                                                    target_size=(224,224),
                                                                    batch_size=32,
                                                                    class_mode="binary")
```


## Tiny VGG Model

```python
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
```

![accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/CNN.png)


## Non-CNN Model

```python
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
```

![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/NCNN.png)


## Non-CNN Model with 25x more node inside layers

```python
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
```

![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/NCNN25x.png)

## CNN model with augmented data and improve structure
```python
model_4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_4 = model_1.fit(train_data_augmented, epochs=10,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
```

![CNNwAug](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/CVnCNN/Binary/Image/CNNwAug.png)

## Model Parameter

| Model             | Parameter | Valid Accuracy |
|-------------------|-----------|----------------|
| CNN               | 31,101| 86.4%          |
| Non-CNN           | 602,141| 50.0%          |
| Non-CNN 25x nodes | 15,063,101| 69.2%          |
| CNN model with augmented data and improve structure      | 82,593| 88.0%          |

## Note
* Make a base model first
* Reduce over fitting with MaxPooling
* Shuffle is important

### Induce overfitting
* Too many conv layer
* Too many conv filter

### Reduce overfitting
* Data augmentation
* Regularization layer
* More data
* Transfer learning

## Layers parameter

### Conv2D

| Parameter   | Description                                                                                 |
|-------------|---------------------------------------------------------------------------------------------|
| Filters     | How many filters in input tensor. ( greater numbers, more complex ) ~ Like node in Dense layer |
| Kernel Size | Shape of filter (greater numbers, greater features )                         |
| Padding     | Hold more info then "same", compress info then "valid"                                      |
| Strides     | Step of filter to walk along the image                                            |

### MaxPool2D

| Parameter | Description                   |
|-----------|-------------------------------|
| Pool Size | How big the window of pooling |