# Binary Classification

## Import
```python
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
```

## 2D decision boundary
```python
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)
    if len(y_pred[0]) > 1:
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        y_pred = np.round(y_pred).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
```

## Loss
``tf.keras.losses.BinaryCrossentropy()``

## Model Experiments

### Model One
```python
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
```

#### Model Structure
![model_1](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_1.png)

#### Loss Curves
![model_1_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_1_loss_curves.png)

#### Decision Boundary
![model_1_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_1_decision_bounday.png)
### Model Two
```python
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])
```

#### Model Structure
![model_2](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_2.png)

#### Loss Curves
![model_2_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_2_loss_curves.png)

#### Decision Boundary
![model_2_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_2_decision_bounday.png)

### Model Three
```python
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
```
#### Model Structure
![model_3](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_3.png)

#### Loss Curves
![model_3_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_3_loss_curves.png)

#### Decision Boundary
![model_3_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_3_decision_bounday.png)

### Model Four
```python
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="linear")
])
```
#### Model Structure
![model_4](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_4.png)

#### Loss Curves
![model_4_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_4_loss_curves.png)

#### Decision Boundary
![model_4_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_4_decision_bounday.png)

### Model Five
```python
model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu")
])
```
#### Model Structure
![model_5](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_5.png)

#### Loss Curves
![model_5_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_5_loss_curves.png)

#### Decision Boundary
![model_5_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_5_decision_bounday.png)

### Model Six
```python
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```
#### Model Structure
![model_6](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_6.png)

#### Loss Curves
![model_6_loss_curves](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_6_loss_curves.png)

#### Decision Boundary
![model_6_decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkClassification/BinaryClassification/Image/model_6_decision_bounday.png)