# Scaling Up

## Dataset

## Tensor

## Check file & preprocess
```python
file.check_file("../Dataset/101_food_classes_10_percent")

train_data_10, test_data_10 = image_data.get_dataset("../Dataset/101_food_classes_10_percent/train",
                                                     "../Dataset/101_food_classes_10_percent/test",
                                                     class_mode="categorical")
```
> [`file`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/file.py), [`image_data`](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/tensor_tool/image_data.py)

## Model

> Functional Api used at this case

### BaseLine
```python
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

x = image_data.augmentation_layer()(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAvgPool2D(name="global_avg_pool_2D")(x)

outputs = tf.keras.layers.Dense(101, activation="softmax", name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_base = model_0.fit(train_data_10,
                           epochs=5,
                           steps_per_epoch=len(train_data_10),
                           validation_data=test_data_10,
                           validation_steps=int(0.5 * len(test_data_10)),
                           callbacks=[callbacks.create_tensorboard_callback("food_101", "BaseLine"),
                                      callbacks.create_model_checkpoint("baseline_checkpoint")])
```
![Loss]()
![Accuracy]()

### Model_1
```python
model_0.load_weights("baseline_checkpoint/checkpoint.ckpt")

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)

model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=["accuracy"])

history_1 = model_0.fit(train_data_10,
                        epochs=10,
                        initial_epoch=history_base.epoch[-1],
                        steps_per_epoch=len(train_data_10),
                        validation_data=test_data_10,
                        validation_steps=int(0.5 * len(test_data_10)),
                        callbacks=[callbacks.create_tensorboard_callback("food_101", "model_1"),
                                   callbacks.create_model_checkpoint("model_1_checkpoint")])
```
![Loss]()
![Accuracy]()

### Downloaded Model
```python
downloaded_model = tf.keras.models.load_model("./06_101_food_class_10_percent_saved_big_dog_model")
result_downloaded = downloaded_model.evaluate(test_data_10)
```

## Performance

### Confusion Matrix
![confusion_matrix]()
### F1-score
![f1-score]()
### Precision
![precision]()
### Recall
![recall]()

### Top 9 wrong prediction
![top9]()
