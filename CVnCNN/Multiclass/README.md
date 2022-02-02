# Big Food Vision

## Dataset

Foods from [Food101](https://www.kaggle.com/kmader/food41)

## Check import files

Use [`check_file`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/file.py) method
from [`tensor_tool`](https://github.com/UncleThree0402/LearningTensorFlow/tree/master/tensor_tool)

```python
tensor_tool.file.check_file("../../Dataset/10_food_classes_all_data")
```

## Create classes names

Use [`get_classes_name`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/file.py) method
from [`tensor_tool`](https://github.com/UncleThree0402/LearningTensorFlow/tree/master/tensor_tool)

```python
tensor_tool.file.get_classes_name("../../Dataset/10_food_classes_all_data/train")
```

## Preprocess Data

### Non-augmented

```python
train_data_gen = ImageDataGenerator(rescale=1. / 255.)

valid_data_gen = ImageDataGenerator(rescale=1. / 255.)

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
```

### Augmented

```python
train_data_gen_augmented = ImageDataGenerator(rescale=1. / 255.,
                                              rotation_range=0.3,
                                              shear_range=0.3,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)

train_date_augmented = train_data_gen_augmented.flow_from_directory(train_dir,
                                                                    target_size=(224, 224),
                                                                    batch_size=32,
                                                                    class_mode="categorical",
                                                                    seed=42)
```

## Model

### Tiny VGG (Without data augmentation)

```python
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
```

### Tiny VGG (With data augmentation)

```python
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
```

### Tiny VGG (With data augmentation & 10 epochs)

```python
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
```

### Self Design CNN (With data augmentation)

```python
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
```

### Difference

| Model                                                 | Parameter | Accuracy |
|-------------------------------------------------------|-----------|----------|
| Tiny VGG(Without data augmentation)                   | 283,920   | 26.00%   |
| Tiny VGG(With data augmentation)                      | 283,920   | 39.80%   |
| Tiny VGG(With data augmentation & 10 epochs)          | 283,920   | 43.36%   |
| Self Design CNN (With data augmentation & 10 epochs)  | 218,585   | 38.48%   |
