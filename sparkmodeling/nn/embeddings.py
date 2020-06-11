import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, Dense, merge, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.models import load_model
from .keras_fixes import dot_mode, cos_mode
import time
import logging


class KerasEmbeddingRegressor(BaseEstimator):
    """ A Keras Embedding regressor wrapped in a sklearn regressor interface.

    Parameters:
    -----------
    vocab_size: int
        The maximum number of workloads to embed.
    dim_embedding: int, optional (default=5)
        The dimension of each embedding vector
    nh: int, optional (default=10)
        The number of 'neurones' (also called hidden units) inside each layer.
        Note: If `pyramid_like` is set to `False`, then the number of neurones
        at each hidden layer's output is the same. If `pyramid_like` is set to
        True, then this number decreases as we go deeper.
    activation: {'linear', 'relu', 'sigmoid', 'tanh', 'hard_sigmoid',
                 'softsign','softplus', 'selu', 'elu', 'softmax'}
                 optional (default='linear')
        The activation used at the output of each layer
    depth: int, optional (default=1)
        The depth represents here the number of hidden fully connected layers
        inside the neural network. Here the embedding layer is not considered
        among those hidden layers.
    pyramid_like: boolean, (default=False)
        If set to False, then the number of units (neurones) in each layer will
        be equal to `nh`
        If set to True, then the number of units will be divided by two as we
        move deeper in the neural network.
    loss: {'MAPE', 'MSE'}, optional (default='MAPE')
        - 'MAPE' is the mean average percentage error
        - 'MSE' is the mean squared error
        The loss function to be minimized during backpropagation.
    optimizer: {'adam', 'adagrad'}, default='adam'
        The optimizer used in the backpropagation.
        Note: currently the only optimizer ready for use is adam
    initial_learning_rate: float, optional (default=0.01)
        The initial learning rate used by the optimizer.
        Ignored if the optimizer is "adam"
    batch_size: int, optional (default=32)
        The size of the mini-batches on which a training step will be done.
    n_epochs: int, optional (default=1000)
        The number of passes over all the data
    early_stopping: boolean, optional (default=False)
        If set to True, it stops fitting when the validation loss stops
        improving even if `n_epochs` hasn't been reached.
    patience: int, optional (default=100)
        Number of epochs with no improvement in the validation loss after which training will be
        stopped. Ignored if early_stopping is set to False.
    l2_reg: float or None, optional (default=None)
        If not None, it adds l2 regularization on the weight matrices
    auto_recovery: boolean, optional (default=True)
        If True, then the regressor will be in 'auto_recovery' mode: it will
        first try to fit on a small number of epochs (=`recovery_epochs`) and
        monitor the loss function, if the loss function doesn't decreases by
        more than 1%, then the regressor will re-initialize the weights and
        check again (for a maximum number of `recovery_trials` trials) to see
        whether for a different initialization it gets better results.
        Once the regressor makes sure that the training loss has decreased,
        it then proceeds and continues fitting until `n_epochs`.

    recovery_trials: int, optional (default=15)
        Number of times to re-initialize the neural network and start fitting
        the regressor from the beginning because the neural network with
        previous initializations failed to drop down the training loss.
        Ignored if auto_recovery==False
    recovery_epochs: int, optional (default=15)
        The number of epochs after which the training loss is checked to see
        if it has dropped or not.
        Ignored if auto_recovery==False
    random_state: int, optional (default=42)
        Random seed to be used.
    """

    def __init__(
            self, vocab_size, len_knob_cols, dim_embedding=5, nh=10,
            activation='linear', depth=1, pyramid_like=False, loss='MAPE',
            optimizer='adam', initial_learning_rate=0.01, batch_size=32,
            n_epochs=1000, early_stopping=False, patience=100, l2_reg=None,
            auto_recovery=True, recovery_trials=15, recovery_epochs=15,
            compat_mode=False,
            random_state=42, model_name=""):
        self.vocab_size = vocab_size
        self.dim_embedding = dim_embedding
        self.nh = nh
        self.activation = activation
        self.depth = depth
        self.pyramid_like = pyramid_like
        self.loss = loss
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.n_epochs = int(n_epochs)
        self.early_stopping = early_stopping
        self.patience = patience
        self.l2_reg = l2_reg
        self.auto_recovery = auto_recovery
        self.recovery_trials = recovery_trials
        self.recovery_epochs = recovery_epochs
        self.random_state = random_state
        self.model_name = model_name  # used for defining different variable scopes
        self.len_knob_cols = len_knob_cols

        self._fitted = False

        self.kill = False
        self.compat_mode = compat_mode

    def init_architecture(self, dim_output=1):
        """Creates the neural network architecture and initialize its weights

        Parameter:
        ----------
        dim_output: int, optional (default=1)
            The dimension of the output. If we're predicting only the latency,
            then dim_output is 1. However, if we want to introduce other
            observations, we need to specify the dimension of the observations
            vector.
        """
        if self.pyramid_like:
            fc_dimensions = [int(self.nh / (2**i)) for i in range(self.depth)]
        else:
            fc_dimensions = [int(self.nh) for i in range(self.depth)]
        fc_dimensions += [dim_output]
        if 0 in fc_dimensions:
            self.kill = True
            return None

        latent_dim = self.dim_embedding
        n_jobs = self.vocab_size
        job_alias = Input((1,), name='job_alias')
        configuration = Input((self.len_knob_cols,), name='configuration_input')
        if self.l2_reg is not None:
            l2_reg = l2(self.l2_reg)
        else:
            l2_reg = None

        workload_layer = Embedding(n_jobs, latent_dim, input_length=1,
                                   name='workload_embedding')
        workload_embedding = Flatten()(workload_layer(job_alias))
        if self.compat_mode:
            from keras.layers import concatenate
            out = concatenate([workload_embedding, configuration])
        else:
            out = merge([workload_embedding, configuration], mode='concat')

        ln = len(fc_dimensions)
        activation = self.activation
        for i, output_size in enumerate(fc_dimensions):
            out = Dense(output_size, activation=activation)(out)
        model = Model(input=[job_alias, configuration], output=out)

        if self.loss == 'MAPE':
            if self.compat_mode:
                div_op = tf.compat.v1.div
            else:
                div_op = tf.div
            model.compile(optimizer=self.optimizer,
                          loss=lambda y_true, y_pred: tf.reduce_mean(
                              div_op(tf.abs(y_pred - y_true), y_true)))
        elif self.loss == "MSE":
            model.compile(optimizer=self.optimizer, loss='MSE')
        else:
            print("Unsupported loss: %s" % str(self.loss))
            raise
        self.keras_model = model

    def fit(self, X, y, log_time=False):
        """Builds the neural network and fits on the training set (X, y)

        Parameters:
        -----------
        X: array-like matrix of shape [n_samples, n_features]
            The training input samples. The first column of X has to be either
            the job id or an alias for the job. The remaining columns must have
            the features: BatchInterval, BlockInterval, Parallelism, InputRate.
            We can also introduce any other features that we find useful for
            the prediction (like polynomial features, etc...)
        y: array-like, shape=[n_samples,]
            The target values (latencies)
        """
        t0 = time.time()
        if np.ndim(X) == 1:
            aliases = X[0]
            aliases = np.reshape(aliases, [1, 1]).astype(int)
            X = X[1:]
            X = np.reshape(X, [1, len(X)])
        else:
            aliases = X[:, 0]
            aliases = np.reshape(aliases, [len(aliases), 1])
            X = X[:, 1:]
        if np.ndim(y) == 1:
            y = np.reshape(y, [len(y), 1])
            dim_output = 1
        elif np.ndim(y) == 0:
            y = np.asarray([[y]])
        else:
            dim_output = np.shape(y)[1]

        repeat = self.recovery_trials
        if self.auto_recovery and not self._fitted:
            while repeat > 0:
                self.init_architecture(dim_output=dim_output)
                if self.kill:
                    return None

                params = {'nb_epoch': self.recovery_epochs,
                          'batch_size': self.batch_size,
                          'shuffle': True, 'verbose': 0}
                if self.early_stopping:
                    es_cb = EarlyStopping(monitor='val_loss', min_delta=0.001,
                                          patience=self.patience)
                    params['validation_split'] = .1
                    params['callbacks'] = [es_cb]
                history = self.keras_model.fit([aliases, X], y, **params)
                losses = history.history['loss']
                if np.abs(losses[-1] - losses[0]) > 0.01:
                    # print("Breaking after: %d" %
                    #       (self.recovery_trials - repeat))
                    break
                repeat -= 1

        elif not self.auto_recovery and not self._fitted:
            self.init_architecture(dim_output=dim_output)
            if self.kill:
                return None
        # Now fit the required number of epochs:
        params = {'nb_epoch': self.n_epochs,
                  'batch_size': self.batch_size,
                  'shuffle': True, 'verbose': 0}
        if self.early_stopping:
            es_cb = EarlyStopping(monitor='val_loss', min_delta=0.001,
                                  patience=self.patience)
            params['validation_split'] = .1
            params['callbacks'] = [es_cb]

        history = self.keras_model.fit([aliases, X], y, **params)
        self._fitted = True
        self.history = history.history

        t_end = time.time()
        fitting_time = t_end - t0
        self._last_fit_duration = fitting_time
        if log_time:
            logging.info(
                "[Embedding Regressor fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))
        return history

    def predict(self, X):
        """Predict latency for X.

        Parameters:
        -----------
        X: array-like matrix of shape [n_samples, n_features]
            The training input samples. The first column of X has to be either
            the job id or an alias for the job.
        """
        if self.kill:
            return None
        aliases = X[:, 0]
        aliases = np.reshape(aliases, [len(aliases), 1])
        X = X[:, 1:]
        try:
            assert self._fitted is True
            y_pred_value = self.keras_model.predict([aliases, X])
            y_pred_value = np.ravel(y_pred_value)
            return y_pred_value
        except AssertionError:
            print("Error: fit has to be called before predict can be invoked")
            raise

    def MAPE(self, X, y):
        """Returns the Mean Average Percentage Error between the prediction & y
        """
        y_pred = self.predict(X)
        error = 100 * np.mean(np.abs(y - y_pred) / y)
        return error

    def score(self, X, y):
        """Returns the opposite of the Mean average percentage error
        """
        if self.kill:
            return -123456789  # flag to say that something is wrong with the architecture
        score = - self.MAPE(X, y)
        return score

    def get_params(self, deep):
        """Parameters getter required by sklearn's gridsearchCV
        """
        params = {}
        params_names = ['vocab_size', 'dim_embedding', 'nh', 'activation',
                        'depth', 'pyramid_like', 'loss',
                        'optimizer', 'initial_learning_rate',
                        'batch_size', 'n_epochs', 'early_stopping',
                        'patience', 'l2_reg', 'auto_recovery',
                        'recovery_trials',
                        'recovery_epochs', 'random_state']
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

    def get_persist_info(self):
        signature_data = {'vocab_size': self.vocab_size,
                          'len_knob_cols': self.len_knob_cols,
                          'dim_embedding': self.dim_embedding,
                          'nh': self.nh,
                          'activation': self.activation,
                          'depth': self.depth,
                          'pyramid_like': self.pyramid_like,
                          'loss': self.loss,
                          'optimizer': self.optimizer,
                          'initial_learning_rate': self.initial_learning_rate,
                          'batch_size': self.batch_size,
                          'n_epochs': self.n_epochs,
                          'early_stopping': self.early_stopping,
                          'patience': self.patience,
                          'l2_reg': self.l2_reg,
                          'auto_recovery': self.auto_recovery,
                          'recovery_trials': self.recovery_trials,
                          'recovery_epochs': self.recovery_epochs,
                          'random_state': self.random_state,
                          'model_name': self.model_name,
                          'compat_mode': self.compat_mode}

        other_data = {"_fitted": self._fitted,
                      "history": self.history,
                      "weights": self.keras_model.get_weights(),
                      "_last_fit_duration": self._last_fit_duration
                      }
        return {'signature': signature_data,
                'other': other_data}

    def clone(self):
        persist_info = self.get_persist_info()
        return KerasEmbeddingRegressor.make_instance(persist_info)

    def persist(self, filepath):
        persist_info = self.get_persist_info()
        np.save(filepath, persist_info)

    def serialize(self, filepath):
        self.persist(filepath)

    @staticmethod
    def make_instance(persist_info):
        instance = KerasEmbeddingRegressor(**persist_info['signature'])
        instance.init_architecture()
        instance.keras_model.set_weights(persist_info['other']['weights'])
        for key in persist_info['other']:
            if key != 'weights':
                setattr(instance, key, persist_info['other'][key])
        return instance

    @staticmethod
    def load_from_file(filepath):
        persist_info = np.load(filepath)[()]
        instance = KerasEmbeddingRegressor.make_instance(persist_info)
        return instance
