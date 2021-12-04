# Scaling

## Import

```python
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
```

## Transformer
```python
 ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)
```

## Fitting the transformer
```python
ct.fit(X_train)
```

## Transform Data
```python
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
```

## Loss & Mae Graph
![LossNMae](https://github.com/UncleThree0402/LearningTensorFlow/blob/master/NeuralNetworkRegression/Scaling/Loss_Mae_plot.png?raw=true)