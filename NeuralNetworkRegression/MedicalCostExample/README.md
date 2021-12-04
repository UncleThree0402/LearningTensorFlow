# Medical Cost Example

## Read Csv
```python
pd.read_csv(link or path)
```

## Get one-hot-encoder
```python
pd.get_dummies(csv)
```

## Process
```python
# Drop 0 for index 1 for column
csv.drop("item", axis=0/1)</pre>
```
## Get Training Set and Test Set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)
```
## Early Stop Callback
Stop while no more further improve
```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
```

## Plot Losses To Epochs
```python
pd.DataFrame(fit.history).plot()
plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()
```

## Loss & Mae Graph
![lossAndMae](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkRegression/MedicalCostExample/Loss_Mae_plot.png)