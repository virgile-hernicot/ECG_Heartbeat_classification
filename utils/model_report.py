import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import itertools


class Report(object):

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def prepare_standardplot(self, xlabel):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_ylabel('categorical cross entropy')
        ax1.set_xlabel(xlabel)
        ax1.set_yscale('log')
        ax2.set_ylabel('accuracy [% correct]')
        ax2.set_xlabel(xlabel)
        return fig, ax1, ax2

    def class_histogram(self):
        plt.hist(self.model.train_labels)
        plt.show()

    def finalize_standardplot(self, fig, ax1, ax2):
        ax1handles, ax1labels = ax1.get_legend_handles_labels()
        if len(ax1labels) > 0:
            ax1.legend(ax1handles, ax1labels)
        ax2handles, ax2labels = ax2.get_legend_handles_labels()
        if len(ax2labels) > 0:
            ax2.legend(ax2handles, ax2labels)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)

    def plot_history(self):
        fig, ax1, ax2 = self.prepare_standardplot('epoch')
        ax1.plot(self.model.history.history['loss'], label="training")
        ax1.plot(self.model.history.history['val_loss'], label="validation")
        ax2.plot(self.model.history.history['acc'], label="training")
        ax2.plot(self.model.history.history['val_acc'], label="validation")
        self.finalize_standardplot(fig, ax1, ax2)
        plt.show()
        return fig

    def f1_score(self):
        return f1_score(self.model.dataset.test_labels, self.model.predictions, average='weighted')

    def accuracy(self):
        return accuracy_score(self.model.dataset.test_labels, self.model.predictions)

    def confusion_matrix(self):
        return confusion_matrix(self.model.dataset.test_labels, self.model.predictions)

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        cm = self.confusion_matrix()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.model.dataset.list_of_classes))
        plt.xticks(tick_marks, self.model.dataset.list_of_classes, rotation=45)
        plt.yticks(tick_marks, self.model.dataset.list_of_classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
