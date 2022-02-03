import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(y_true, y_preds, classes=None, figsize=(10, 10), text_size=20):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_preds)

    # Get Normalize
    cm_normalize = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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

    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    plt.savefig("./Image/confusion_matrix.png")


def plot_classification_report(y_true, y_pred, class_names, fig_size=(12, 25)):
    classification_result_dit = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    f1_scores = {}
    recall = {}
    precision = {}
    for k, v in classification_result_dit.items():
        if k == "accuracy":
            break
        else:
            f1_scores[class_names[int(k)]] = v["f1-score"]
            precision[class_names[int(k)]] = v["precision"]
            recall[class_names[int(k)]] = v["recall"]

    # F1-scores
    f1_scores = pd.DataFrame({"class_name": list(f1_scores.keys()),
                              "f1-score": list(f1_scores.values())}).sort_values("f1-score", ascending=True)
    # Precision
    precision = pd.DataFrame({"class_name": list(precision.keys()),
                              "precision": list(precision.values())}).sort_values("precision", ascending=True)
    # Recall
    recall = pd.DataFrame({"class_name": list(recall.keys()),
                           "recall": list(recall.values())}).sort_values("recall", ascending=True)

    # F1-scores
    plt.figure(figsize=fig_size)
    plt.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
    plt.yticks(range(len(f1_scores)), f1_scores["class_name"].values)
    plt.title("f1-score of model")
    plt.xlabel("f1-score")
    plt.ylabel("class_name")
    plt.savefig("./Image/f1-score.png")

    # Precision
    plt.figure(figsize=fig_size)
    plt.barh(range(len(precision)), precision["precision"].values)
    plt.yticks(range(len(precision)), precision["class_name"].values)
    plt.title("precision of model")
    plt.xlabel("precision")
    plt.ylabel("class_name")
    plt.savefig("./Image/precision.png")

    # Recall
    plt.figure(figsize=fig_size)
    plt.barh(range(len(recall)), recall["recall"].values)
    plt.yticks(range(len(recall)), recall["class_name"].values)
    plt.title("recall of model")
    plt.xlabel("recall")
    plt.ylabel("class_name")
    plt.savefig("./Image/recall.png")
