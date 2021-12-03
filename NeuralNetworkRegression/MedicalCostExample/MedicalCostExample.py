import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

insurance_one_hot = pd.get_dummies(insurance)

X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

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

history = insurance_model.fit(X_train, y_train, epochs=200, verbose=1, callbacks=[callback])

insurance_model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()
