# Evaluation

## Visualise
```python
def plot_pred(prediction,
              title,
              training_data=X_train,
              training_label=y_train,
              testing_data=X_test,
              testing_label=y_test,):
    plt.figure(figsize=(10, 7))
    plt.scatter(training_data, training_label, c="b", label="Training")
    plt.scatter(testing_data, testing_label, c="g", label="Testing")
    plt.scatter(testing_data, prediction, c="r", label="Prediction")
    plt.legend()
    plt.title(title)
    plt.show()
```

## Three Set
**Training Set** - model learn from **70-80%**<br>
**Validation Set** - model get tuned one **10-15%**<br>
**Test Set** - model get evaluated from **10-15%**<br>


## Parameter
**Total parameter** : total number of parameter of model<br>
**Trainable parameter** : Parameter(s) (patterns) which can update<br>
**Non-trainable parameter** : Parameter that can't update (Already trained)<br>

## Plot_Model
```python
plot_model(model=model)
```

## Error

### MAE
```python
tf.metrics.mean_absolute_error(y_true, y_pred)
```

### MSE
```python
tf.metrics.mean_squared_error(y_true, y_pred)
```