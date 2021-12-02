# Evaluation

### Visualise
<pre>
Plot them out</pre>

### Three Set
<pre>
Training Set - model learn from 70-80 %
Validation Set - model get tuned one 10-15 %
Test Set - model get evaluated from 10-15 %
</pre>

### Parameter
<pre>
Total parameter : total number of parameter of model
Trainable parameter : Parameter(s) (patterns) which can update
Non-trainable parameter : Parameter that can't update (Already trained)</pre>

### Plot_Model
<pre>
plot_model(model=model)</pre>

### Error
<pre>
mae : tf.metrics.mean_absolute_error(y_true, y_pred)
mse : tf.metrics.mean_squared_error(y_true, y_pred)</pre>