import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt


class DataLoader(object):

    def __init__(self, config):
        self.config = config

        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

        # Training Dataset's Data and Lables.
        self.train_data = np.array([])
        self.train_labels = np.array([])

        # Testing Dataset's Data and Lables.
        self.test_data = np.array([])
        self.test_labels = np.array([])

        # Total number of Class-Labels.
        self.no_of_classes = 0
        # Class Label List.
        self.list_of_classes = []

        # One-hot Encoded Label Vector.
        self.train_label_one_hot = np.array([])
        self.test_label_one_hot = np.array([])

        self.load_dataset()

        self.prepreprocess()

        self.print_dataset_infos()

        if self.config.config_namespace.preprocess_verbose:
            self.preprocess_dataset()
            self.labels_histogram()

        return

    def print_dataset_infos(self):
        # Number and shape of dataset
        print("Training data shape: (Data, Labels)", self.train_data.shape, self.train_labels.shape)
        print("Testing data shape: (Data, Labels)", self.test_data.shape, self.test_labels.shape)

        # Number of class labels and their list.
        print("Total number of Classes in the dataset: ", self.no_of_classes)
        print("The ", self.no_of_classes, " Classes of the dataset are: ", self.list_of_classes)

    def load_dataset(self):
        raise NotImplementedError

    def prepreprocess(self):
        self.train_data = np.array(self.df_train[list(range(self.config.config_namespace.class_index))].values)[
            ..., np.newaxis]
        self.train_labels = np.array(self.df_train[self.config.config_namespace.class_index].values).astype(np.int8)

        self.test_labels = np.array(self.df_test[self.config.config_namespace.class_index].values).astype(np.int8)
        self.test_data = np.array(self.df_test[list(range(self.config.config_namespace.class_index))].values)[
            ..., np.newaxis]

        self.train_label_one_hot = to_categorical(self.train_labels)
        self.test_label_one_hot = to_categorical(self.test_labels)

        self.list_of_classes = np.unique(self.train_labels, return_counts=False)
        self.no_of_classes = len(self.list_of_classes)

    def labels_histogram(self):
        plt.hist(self.train_labels)
        plt.show()

    def preprocess_dataset(self):
        raise NotImplementedError
