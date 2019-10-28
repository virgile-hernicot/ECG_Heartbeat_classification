from base.data_loader_base import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


class PtbdbLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)

    def load_dataset(self):
        df_1 = pd.read_csv(self.config.config_namespace.train_data_path, header=None)
        df_2 = pd.read_csv(self.config.config_namespace.test_data_path, header=None)

        df = pd.concat([df_1, df_2])
        self.df_train, self.df_test = train_test_split(df, test_size=self.config.config_namespace.test_size,
                                                       stratify=df[self.config.config_namespace.class_index])

    def preprocess_dataset(self):
        return
