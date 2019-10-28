from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
import numpy as np
import time
from tensorflow.keras.models import Sequential


class BaseModel(object):

    def __init__(self, config, dataset):

        self.config = config
        # Training and testing datasets.
        self.dataset = dataset

        # ConvNet model.
        self.model = Sequential()

        # History object, holds training history.
        self.history = History()

        # Saved model path.
        self.saved_model_path = self.config.config_namespace.saved_model_path

        # Checkpoint for ConvNet model.
        self.checkpoint = ModelCheckpoint(self.saved_model_path, monitor='val_acc',
                                          verbose=self.config.config_namespace.checkpoint_verbose, save_best_only=True,
                                          mode='max')
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', restore_best_weights=True)

        # Callbacks list.
        self.callbacks_list = [self.checkpoint, self.early_stopping]

        # Evaluation scores.
        self.scores = []

        # Training time.
        self.train_time = 0

        # Predicted class labels.
        self.predictions = np.array([])
        self.predictions_one_hot = np.array([])

        # Construct the ConvNet model.
        self.define_model()

        # Configure the ConvNet model.
        self.compile_model()

        # Train the ConvNet model using testing dataset.
        self.fit_model()

        # Evaluate the ConvNet model using testing dataset.
        self.evaluate_model()

        # Predict the class labels of testing dataset.
        self.predict()

    def define_model(self):
        """
        Constructs the ConvNet model.
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to add layers to the network.
        raise NotImplementedError

    def compile_model(self):
        """
        Configure the model.
        :return none
        :raises none
        """

        self.model.compile(loss=self.config.config_namespace.compile_loss,
                           optimizer=self.config.config_namespace.compile_optimizer,
                           metrics=[self.config.config_namespace.compile_metrics1])

    def fit_model(self):
        """
        Train the model.
        :return none
        :raises none
        """

        start_time = time.time()

        if self.config.config_namespace.save_model == "true":
            print("Training phase under progress, trained ConvNet model will be saved at path", self.saved_model_path,
                  " ...\n")
            self.history = self.model.fit(x=self.dataset.train_data,
                                          y=self.dataset.train_label_one_hot,
                                          batch_size=self.config.config_namespace.batch_size,
                                          epochs=self.config.config_namespace.num_epochs,
                                          callbacks=self.callbacks_list,
                                          verbose=self.config.config_namespace.fit_verbose,
                                          validation_data=(self.dataset.test_data, self.dataset.test_label_one_hot))
        else:
            print("Training phase under progress ...\n")
            self.history = self.model.fit(x=self.dataset.train_data,
                                          y=self.dataset.train_label_one_hot,
                                          batch_size=self.config.config_namespace.batch_size,
                                          epochs=self.config.config_namespace.num_epochs,
                                          verbose=self.config.config_namespace.fit_verbose,
                                          validation_data=(self.dataset.test_data, self.dataset.test_label_one_hot))

        end_time = time.time()

        self.train_time = end_time - start_time
        print("The model took %0.3f seconds to train.\n" % self.train_time)

    def evaluate_model(self):
        """
        Evaluate the model.
        :return none
        :raises none
        """

        self.scores = self.model.evaluate(x=self.dataset.test_data, y=self.dataset.test_label_one_hot,
                                          verbose=self.config.config_namespace.evaluate_verbose)

        print("Test loss: ", self.scores[0])
        print("Test accuracy: ", self.scores[1])

    def predict(self):
        """
        Predicts the class labels of unknown data.
        :param none
        :return none
        :raises NotImplementedError: Exception: Implement this method.
        """

        # Implement this method in the inherited class to predict the class-labels of unknown data.
        raise NotImplementedError

    def save_model(self):
        """
        Saves the ConvNet model to disk in h5 format.
        :return none
        """

        if self.model is None:
            raise Exception("ConvNet model not configured and trained !")

        self.model.save(self.saved_model_path)
        print("ConvNet model saved at path: ", self.saved_model_path, "\n")

    def load_model(self):
        """
        Loads the saved model from the disk.
        :return none
        :raises NotImplementedError: Implement this method.
        """

        if self.model is None:
            raise Exception("ConvNet model not configured and trained !")

        self.model.load_weights(self.saved_model_path)
        print("ConvNet model loaded from the path: ", self.saved_model_path, "\n")
