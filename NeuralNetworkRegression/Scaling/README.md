# Scaling

### Import
<pre>
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder</pre>

### Transformer
<pre>
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)</pre>

### Fitting & Transform
<pre>
//Fitting
ct.fit(X_train)

//Transform
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)</pre>