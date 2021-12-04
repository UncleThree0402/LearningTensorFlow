# Sample Regression

## Init model
```python
model = tf.keras.Sequential([ # Layers
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

## Compiling
```python
model.compile(loss=tf.keras.losses.mae , # Loss
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # Optimizer
              metrics=["mae"])
```

## Fitting model
```python
model.fit(tf.expand_dims(X, axis=-1), y, epochs=500) # Train
```

## Predict

```python
print(model.predict([10]))
```