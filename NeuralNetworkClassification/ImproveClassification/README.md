# Improve Classification

## Tweak Learning Rate

```python
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 20))
```

## Different Metrics
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report