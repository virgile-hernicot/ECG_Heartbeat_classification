from base.model_base import BaseModel
from tensorflow.keras.layers import Convolution1D, Dropout, MaxPool1D, Flatten, Dense
from tensorflow.nn import relu, softmax
import numpy as np


class ConvModel(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        return

    def define_model(self):
        self.model.add(Convolution1D(self.config.config_namespace.convolution_dim,
                                     kernel_size=self.config.config_namespace.convolution_kernel_size,
                                     activation=relu, padding='valid',
                                     input_shape=(self.dataset.train_data.shape[1], self.dataset.train_data.shape[2])))
        self.model.add(MaxPool1D(pool_size=self.config.config_namespace.max_pool_size))
        self.model.add(Dropout(self.config.config_namespace.dropout_rate))
        self.model.add(Flatten())
        self.model.add(Dense(self.config.config_namespace.intermediate_dim, activation=relu))
        self.model.add(Dense(self.dataset.train_label_one_hot.shape[1], activation=softmax))

    def predict(self):
        """
        Predict the class labels of testing dataset.
        :return none
        :raises none
        """

        self.predictions_one_hot = self.model.predict(x=self.dataset.test_data,
                                                      verbose=self.config.config_namespace.predict_verbose)
        self.predictions = self.predictions_one_hot.argmax(axis=-1).astype(np.int8)
