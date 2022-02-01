import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_preds, classes=None, figsize=(10,10), text_size=20):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_preds)

    # Get Normalize
    cm_normalize = cm.astype("float") / cm.sum(axis=1) [: , np.newaxis]

    # Classes
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
           xlabel="Predicted Table",
           ylabel="True Table",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set Label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_normalize[i , j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i , j] > threshold else "black",
                 size=text_size)
    plt.show()