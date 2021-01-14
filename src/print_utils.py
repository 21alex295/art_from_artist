"""Utils file for printing figures."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def print_cm(cm, unique_labels):
    """Print the confusion matrix."""
    sns.heatmap(cm, annot=True, xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.show()


def print_metrics(accuracy, precision):
    """Print the accuracy and precision metrics in a figure."""
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(accuracy, marker="o", color="r", label="accuracy")
    ax.plot(precision, marker="o", color="b", label="precision")
    ax.legend()
    ax.set_title("Random Forest")
    ax.set_xlabel("Test Size")
    ax.grid(True, linestyle="--")
    plt.show()


def print_bars(accuracy, precision):
    """Print a bars graphic for the feature importance."""
    labels_bar = ["GIST", "HOG", "Color Hists", "Texton", "Tiny", "LBP"]
    x = np.arange(len(labels_bar))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()

    ax.bar(x - width/2, accuracy, width, label='Accuracy')
    ax.bar(x + width/2, precision, width, label='Precision')
    ax.set_title('Feature Importances')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.legend()
    plt.show()
