# Fine tuning

## How
* unfroze some layers of transferred model
* lower learning rate
* Add dataset

## Data

Foods from [Food101](https://www.kaggle.com/kmader/food41)

## Check File

```python
file.check_file("../Dataset/10_food_classes_10_percent")
file.check_file("../Dataset/10_food_classes_1_percent")
file.check_file("../Dataset/10_food_classes_all_data")
```

> [`file`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/file.py)

## Preprocess

This time we use different method

```python
train_data_10, test_data_10 = image_data.get_dataset("../Dataset/10_food_classes_10_percent/train",
                                                     "../Dataset/10_food_classes_10_percent/test",
                                                     class_mode="categorical")

train_data_1, test_data_1 = image_data.get_dataset("../Dataset/10_food_classes_1_percent/train",
                                                   "../Dataset/10_food_classes_1_percent/test",
                                                   class_mode="categorical")

train_data_all, test_data_all = image_data.get_dataset("../Dataset/10_food_classes_all_data/train",
                                                       "../Dataset/10_food_classes_all_data/test",
                                                       class_mode="categorical")
```

> Return Dataset | [`image_data`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/image_data.py)

## Model

> Functional Api used at this case

### BaseLine

```python
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

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
```

![Loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/0_loss.png)
![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/0_ac.png)

### Model_1

```python
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
```

![Loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/1_loss.png)
![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/1_ac.png)

### Model_2

```python
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
```

![Loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/2_loss.png)
![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/2_ac.png)

### Model_3

```python
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
```

![Loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/3_loss.png)
![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/3_ac.png)

### Model_4

```python
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
```

![Loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/4_loss.png)
![Accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/fine_tuning/Image/4_ac.png)

## Different
**5 epochs**

| Model    | Info                                        | Dataset       | Lost   | Accuracy |
|----------|---------------------------------------------|---------------|--------|----------|
| BaseLine | EfficientNetB0 without tuning               | 10% 10_foods  | 0.6294 | 83.16%   |
| model_1  | EfficientNetB0 without tuning               | 1% 10_foods   | 1.8151 | 47.96%   |
| model_2  | EfficientNetB0 without tuning               | 10% 10_foods  | 0.6976 | 81.08%   |
| model_3  | EfficientNetB0 with tuning (last 10 layers) | 10% 10_foods  | 0.4956 | 83.08%   |
| model_4  | EfficientNetB0 with tuning (last 10 layers) | 100% 10_foods | 0.3342 | 89.12%   |