# MulticlassClassification

## Import Fashion mnist

```python
from tensorflow.keras.datasets import fashion_mnist
(train_data, train_labels) , (test_data, test_labels) = fashion_mnist.load_data()
```

### Check how data look like
```python
# Check raw data
print(f"Train Data : \n {train_data[0]} \n")
print(f"Train Label : \n {train_labels[0]} \n")

# Shape of Data
print(train_data[0].shape, train_labels[0].shape)

# Show image
plt.imshow(train_data[7], cmap=plt.cm.gray)
plt.show()
```

### Label
```python
image_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                 "Ankle Boot"]
```

### Test
```python
index_image = 15
plt.imshow(train_data[index_image], cmap=plt.cm.binary)
plt.show()
print(image_classes[train_labels[index_image]])
```

### Plot random graph
```python
plt.figure(figsize=(4,4))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    random_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[random_index], cmap=plt.cm.binary)
    plt.title(image_classes[train_labels[random_index]])
    plt.axis(False)
plt.show()
```

## Model
```python
tf.random.set_seed(42)
```

### Info
Loss function : CategoricalCrossentropy (One_hot encoded) / SparesCategoricalCrossentropy (With out one_hot encoding)<br>
Optimizer : Adam<br>
Metrics : accuracy

### Model without normalization
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax"),
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(train_data, tf.one_hot(train_labels, depth=10),epochs=10,
          validation_data=(test_data, tf.one_hot(test_labels, depth=10)))
```

### Model with normalization
#### How to normalize
```python
# Find min & max
print(train_data.min(), train_data.max())
```

#### Normalize
```python
# Normalize
train_data_nor = train_data / 255.0
test_data_nor = test_data / 255.0
```

#### Model
```python
model_norm = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model_norm.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])

history_norm = model_norm.fit(train_data_nor, tf.one_hot(train_labels, depth=10), epochs=10,
                              validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)))
```

### Find best learning rate

#### Callback
```python
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 20))
```

#### Model
```python
model_lr = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model_lr.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=["accuracy"])

history_lr = model_lr.fit(train_data_nor, tf.one_hot(train_labels, depth=10), epochs=100,
                              validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)),
                          callbacks=[lr_scheduler])
```

#### Result
![lr_loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/dev/NeuralNetworkClassification/MulticlassClassification/Image/lr_loss.png)

### Model after tweak
```python
model_tweak = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=train_data[0].shape),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_tweak.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=["accuracy"])

history_tweak = model_tweak.fit(train_data_nor, tf.one_hot(train_labels, depth=10),
                                epochs=20,
                                validation_data=(test_data_nor, tf.one_hot(test_labels, depth=10)))
```

### Result of all model
![result](https://github.com/UncleThree0402/LearningTensorFlow/blob/dev/NeuralNetworkClassification/MulticlassClassification/Image/accuracy.png)
![confusion](https://github.com/UncleThree0402/LearningTensorFlow/blob/dev/NeuralNetworkClassification/MulticlassClassification/Image/confusionmatrix.png)