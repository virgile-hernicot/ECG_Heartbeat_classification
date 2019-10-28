from base.data_loader_base import DataLoader
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


class MitbihLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)

    def load_dataset(self):
        self.df_train = pd.read_csv(self.config.config_namespace.train_data_path, header=None)
        self.df_test = pd.read_csv(self.config.config_namespace.test_data_path, header=None)

    def balance_data(self):
        values, count = np.unique(self.train_labels, return_counts=True)
        average_other = int(np.average(count[1:]))
        x_first = self.train_data[self.train_labels == 0]
        y_first = self.train_labels[self.train_labels == 0]
        index = np.random.choice(x_first.shape[0], average_other, replace=False)
        x_train_new = self.train_data[self.train_labels > 0]
        y_train_new = self.train_labels[self.train_labels > 0]
        self.train_data = np.concatenate([x_train_new, x_first[index]])
        self.train_labels = np.concatenate([y_train_new, y_first[index]])
        self.train_label_one_hot = to_categorical(self.train_labels)

    def preprocess_dataset(self):
        if self.config.config_namespace.preprocess_verbose:
            self.labels_histogram()
        self.balance_data()
        return
