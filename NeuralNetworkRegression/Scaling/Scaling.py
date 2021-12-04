import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# Transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Fit
ct.fit(X_train)

# Transform
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

# Stop Callback
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

history = insurance_model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[callback])

insurance_model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()
