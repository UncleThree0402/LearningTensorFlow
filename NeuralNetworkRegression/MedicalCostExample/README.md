# Medical Cost Example

### Read Csv
<pre>
pd.read_csv(link or path)
</pre>

### Get one-hot-encoder
<pre>
pd.get_dummies(csv)</pre>

### Process
<pre>
//Drop 0 for index 1 for column
csv.drop("item", axis=0/1)</pre>

### Get Training Set and Test Set
<pre>
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)</pre>

### Early Stop Callback
Stop while no more further improve
<pre>
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)</pre>

### Plot Losses To Epochs
<pre>
pd.DataFrame(fit.history).plot()
plt.xlabel("epochs")
plt.ylabel("losses")
plt.show()</pre>