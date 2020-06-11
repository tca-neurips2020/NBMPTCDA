from keras.layers import Input, Dropout
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import logging
import time


class NNregressor:
    def __init__(self, n_hidden_layers=1, nh=10, early_stopping=True,
                 dropout=0, nb_epochs=1000, random_state=42,
                 override_initial_weights=None, with_calibration=False,
                 input_shape=(14, ),
                 activation='relu', patience=100,
                 with_tensorboard=False, verbose=0,
                 learning_rate=0.001, loss='mape', v1_compat_mode=False,
                 keras_2=False):
        """
        A Neural Network Regressor following sklearn's API (fit, predict,
        score, etc...)
        v1_compat_mode: boolean
            Set to true if using tensorflow v2.0 and needs to use compat operators from v1.
        keras_2: boolean
            Set to true if using keras 2 (signature changed when fitting...)
        """

        self.n_hidden_layers = n_hidden_layers
        self.input_shape = input_shape
        self.nh = nh
        self.early_stopping = early_stopping
        self.nb_epochs = nb_epochs
        self.dropout = dropout

        self.override_initial_weights = override_initial_weights
        self.with_calibration = with_calibration
        self.fitted = False
        self.activation = activation
        self.patience = patience
        self.with_tensorboard = with_tensorboard
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.loss = loss
        if self.with_calibration:
            self.my_offset = 0

        self.v1_compat_mode = v1_compat_mode
        self.keras_2 = keras_2

    def clone(self):
        """
        Returns a cloned version of this neural network regressor.
        """
        copy = NNregressor()
        for key, value in self.__dict__.items():
            if key != 'model':
                setattr(copy, key, value)
        copy._create_architecture()
        copy.model.set_weights(copy.trained_weights)
        return copy

    def persist(self, filepath):
        """
        Serialize the current Neural Network with its weights to disk

        filepath: str
            filepath containing the filename with the npy extension. If
            no extension is provided, then automatically the file will
            have npy extension.
        """
        params = get_persist_info()
        np.save(filepath, params)

    def get_persist_info(self):
        params = {}
        for key, value in self.__dict__.items():
            if key != 'model':
                params[key] = value
        return params

    def load_trained_regressor(self, filename):
        params = np.load(filename)[()]
        for key, value in params.items():
            setattr(self, key, value)
        self._create_architecture()
        self.model.set_weights(self.trained_weights)

    def load_trained_regressor_(self, params):
        logging.debug("Setting values...")
        for key, value in params.items():
            setattr(self, key, value)
        logging.debug("Creating architecture")
        self._create_architecture()
        logging.debug("Setting weights")
        self.model.set_weights(self.trained_weights)
        logging.debug("Done setting weights for the model")

    def init_architecture(self):
        input = Input(shape=self.input_shape)
        o1 = Dense(self.nh, activation=self.activation)(input)
        if self.dropout > 0:
            o1 = Dropout(self.dropout)(o1)
        last = o1
        for i in range(self.n_hidden_layers - 1):
            o2 = Dense(self.nh, activation=self.activation)(last)
            if self.dropout > 0:
                o2 = Dropout(self.dropout)(o2)
            last = o2
        output = Dense(1)(last)
        if self.keras_2:
            model = Model(inputs=input, outputs=output)
        else:
            model = Model(input=input, output=output)
        optimizer = Adam(lr=self.learning_rate)
        if self.v1_compat_mode:
            div_op = tf.compat.v1.div
        else:
            div_op = tf.div
        if self.loss == "mape":
            model.compile(optimizer=optimizer, loss=lambda y_true,
                          y_pred: tf.reduce_mean(
                              div_op(tf.abs(y_pred - y_true), y_true)))
        elif self.loss == "mean_squared_error":
            model.compile(optimizer=optimizer, loss=self.loss)
        else:
            print("Unknown loss")
            import sys
            sys.exit(0)

        self.model = model

    def _create_architecture(self):
        input = Input(shape=self.input_shape)
        o1 = Dense(self.nh, activation=self.activation)(input)
        if self.dropout > 0:
            o1 = Dropout(self.dropout)(o1)
        last = o1
        for i in range(self.n_hidden_layers - 1):
            o2 = Dense(self.nh, activation=self.activation)(last)
            if self.dropout > 0:
                o2 = Dropout(self.dropout)(o2)
            last = o2
        output = Dense(1)(last)
        if self.keras_2:
            model = Model(inputs=input, outputs=output)
        else:
            model = Model(input=input, output=output)
        optimizer = Adam(lr=self.learning_rate)
        if self.v1_compat_mode:
            div_op = tf.compat.v1.div
        else:
            div_op = tf.div
        if self.loss == "mape":
            model.compile(
                optimizer=optimizer, metrics=['mse'],
                loss=lambda y_true, y_pred: tf.reduce_mean(
                    div_op(tf.abs(y_pred - y_true),
                           y_true)))
        elif self.loss == "mean_squared_error":
            model.compile(optimizer=optimizer, loss=self.loss)
        else:
            print("Unknown loss")
            import sys
            sys.exit(0)
        self.model = model

    def fit(self, X, y, log_time=False):
        """
        Fits the Neural Network Regressor on the given inputs
        """
        t0 = time.time()
        callbacks = None
        if not self.fitted:
            self.fitted = True
            self._create_architecture()
            if self.override_initial_weights is not None:
                self.model.set_weights(self.override_initial_weights)
            self.initial_weights = self.model.get_weights()
            if self.early_stopping:
                es_cb = EarlyStopping(patience=self.patience)
                callbacks = [es_cb]
                if self.with_tensorboard:
                    tb_cb = TensorBoard()
                    callbacks.append(tb_cb)
                if self.keras_2:
                    history = self.model.fit(
                        X, y, epochs=self.nb_epochs, verbose=self.verbose,
                        callbacks=callbacks, validation_split=0.1)
                else:
                    history = self.model.fit(
                        X, y, nb_epoch=self.nb_epochs, verbose=self.verbose,
                        callbacks=callbacks, validation_split=0.1)
                self.history_ = history.history
            else:
                if self.with_tensorboard:
                    callbacks = [tb_cb]
                if self.keras_2:
                    history = self.model.fit(
                        X, y, epochs=self.nb_epochs, callbacks=callbacks,
                        verbose=self.verbose)
                else:
                    history = self.model.fit(
                        X, y, nb_epoch=self.nb_epochs, callbacks=callbacks,
                        verbose=self.verbose)
                self.history_ = history.history
            self.trained_weights = self.model.get_weights()
        else:
            # incremental fit
            self.model.fit(X, y, nb_epoch=self.nb_epochs, verbose=self.verbose)

        t_end = time.time()
        fitting_time = t_end - t0
        self._last_fit_duration = fitting_time
        if log_time:
            logging.info(
                "[NN fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))

    def calibrate(self, X_calib, y_calib):
        """
        Calibrate the regressor by adjusting the offset on new given input data
        """
        if self.with_calibration:
            self.my_offset = 0
            y_pred = self.predict(X_calib)
            self.my_offset = np.mean(y_calib - y_pred)
        else:
            raise Exception("Unsupported Operation")

    def predict(self, X):
        """
        Predicts the values for the given input X
        """
        predictions = np.ravel(self.model.predict(X))
        if self.with_calibration:
            predictions_adjusted = predictions + self.my_offset
            predictions = list(predictions)
            predictions_adjusted = list(predictions_adjusted)

            to_return = list(map(lambda x, y: x if x > 0 else y,
                                 predictions_adjusted, predictions))
            return np.asarray(to_return)
        return predictions

    def MAPE(self, X, y):
        """
        Returns the Mean Absolute Percentage Error over the given inputs and
        labels
        """

        y_pred = self.predict(X)
        error = 100 * np.mean(np.abs(y - y_pred) / y)
        return error

    def score(self, X, y):
        """
        Returns a score (useful for sklearn gridsearchCV)

        Inputs: same as MAPE
        """

        return -self.MAPE(X, y)

    def get_params(self, deep):
        """Parameters getter required by sklearn's gridsearchCV
        """
        params = {}
        params_names = ['n_hidden_layers', 'nh', 'early_stopping',
                        'nb_epochs', 'dropout', 'random_state']
        for key, value in self.__dict__.items():
            if key in params_names:
                params[key] = value
        return params

    def set_params(self, **params):
        """Parameters setter required by sklearn's gridsearchCV
        """
        for key in params:
            setattr(self, key, params[key])
        return self
