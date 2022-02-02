import matplotlib.pyplot as plt


def plot_loss_curves(history, model_name=""):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(history.history["loss"]))

    plt.plot(epochs, loss, label="training_loss", marker="o")
    plt.plot(epochs, val_loss, label="validation_loss", marker="o")
    plt.title(model_name + "Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def plot_accuracy_curves(history, model_name=""):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["accuracy"]))

    plt.plot(epochs, accuracy, label="training_accuracy", marker="o")
    plt.plot(epochs, val_accuracy, label="validation_accuracy", marker="o")
    plt.title(model_name + "Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
