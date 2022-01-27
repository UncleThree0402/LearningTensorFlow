# Improve Classification

## Tweak Learning Rate

### Find the lowest loss rate of the model
```python
# Using Exponential function 
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 20))
# Put in to callbacks function
history = model.fit(train_X, train_y, epochs=100, verbose=1, callbacks=[lr_scheduler])
```

### Plot learning_rate vs loss
```python
plt.figure(figsize=(10,7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
plt.show()
```

![Learning_rate_loss](https://github.com/UncleThree0402/LearningTensorFlow/blob/readme/NeuralNetworkClassification/ImproveClassification/image/learning_rate_loss.png)

### Tweak, compile and fit

#### Accuracy
![accuracy](https://github.com/UncleThree0402/LearningTensorFlow/blob/readme/NeuralNetworkClassification/ImproveClassification/image/model_accuracy_curves.png)

#### Decision Boundary
![decision_boundary](https://github.com/UncleThree0402/LearningTensorFlow/blob/readme/NeuralNetworkClassification/ImproveClassification/image/model_decision_boundary.png)

### Confusion Matrix
> To Check True / False positive or negative

![confusion_matrix](https://github.com/UncleThree0402/LearningTensorFlow/blob/readme/NeuralNetworkClassification/ImproveClassification/image/confusion_matrix.png)

## Different Metrics
* Accuracy 
* Precision (Less false positive)
* Recall (Less false negative)
* F1-score
* Confusion Matrix
* Classification Report