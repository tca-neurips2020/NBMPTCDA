import tensorflow as tf
import numpy as np
from .saver import Saver
from sklearn.utils import shuffle
import os
from ..common.utils import train_test_split_, identity_tensor
import logging
import time


def weight_variable(shape, trainable=True, init_std=0.1):
    # tensorflow 2.0; 1.x counterpart is tf.truncated_normal
    initial = tf.random.truncated_normal(shape, stddev=init_std)
    return tf.Variable(initial, trainable=trainable)


def bias_variable(shape, trainable=True, init=0.1):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, trainable=trainable)


class FancyAutoEncoder:
    def __init__(self, n_iter, hidden_layer_sizes, activations,
                 initial_learning_rate, solver='Adam', batch_size=32,
                 random_state=10, early_stopping=False, patience=10,
                 validation_set=None, lamda=1e-1, knob_cols=None,
                 auto_refit=True, max_refit_attempts=10):
        """
        This is a modified version of the AutoEncoder imitating interfaces
        provided by scikit-learn for unsupervised learning (fit, transform)

        The main modification brought to this AutoEncoder is a new loss
        function that focus on reconstructing another input. So such
        an AutoEncoder will have 2 types of input:
        1) It's traditional input which should be fed to the encoder layer
        2) Another input which we try to approximate in the bottleneck layer.

        lamda: coefficient multiplying the configuration approximation term
        auto_refit: whether to autorefit if centroids are vanishing
        max_refit_attempts: maximum number of attempts for refitting...

        """

        if knob_cols is None:
            # FIXME
            pass
        self.knob_cols = knob_cols

        self.n_iter = n_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = activations
        l = int(len(self.activations) / 2)

        self.hidden_layer_sizes[int(
            len(self.hidden_layer_sizes) / 2)] += len(self.knob_cols)

        self.solver = solver
        self.random_state = random_state
        self.initial_learning_rate = initial_learning_rate
        self._fitted = False
        self.batch_size = batch_size
        self.train_encodings = None

        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_set = validation_set

        self.lamda = lamda  #
        self.centroids = None

        self.altered_centroids = None
        self._last_fit_duration = None

        self.refit_attempt = None
        self.auto_refit = auto_refit  # auto refit on failure
        self.max_refit_attempts = max_refit_attempts

        np.random.seed(random_state)

        try:
            assert len(hidden_layer_sizes) % 2 == 1
            self.__create_activation_map()
        except AssertionError:
            print("Error: the length of hidden_layer_sizes must be odd")
            raise

    def _compute_centroids(self, encodings, labels):
        """Computes the centroids of encodings given its labels...

        ! Note: centroids are indexed by job alias not by job id...
        """
        counts = {}
        centroids = {}

        encodings = encodings.copy()

        for i, encoding in enumerate(encodings):
            key = int(labels[i])
            if key in centroids:
                centroids[key] += encoding
                counts[key] += 1
            else:
                centroids[key] = encoding
                counts[key] = 1
        for key in centroids:
            centroids[key] /= counts[key]
        self.centroids = centroids

    def fake_fit(self, nn_weights):
        self._architecture = self.__build_nn_architecture(
            self.dim_Y, self.hidden_layer_sizes)
        self._placeholders, self._weights, self._biases, \
            self._outputs = self._architecture
        encoding_vector_index = int(len(self.hidden_layer_sizes) / 2)
        self._saver = Saver(
            self._weights[:encoding_vector_index + 1] + self._biases)
        self._saver.best_params = nn_weights
        self._fitted = True
        logging.warning(
            "fake_fit has been called without calling restore_weights from saver")

    def fit(self, X, debug=False,
            centroids_strategy='all',
            X_shared=None,
            log_time=False,
            refit_attempt=0):
        """
        X: numpy array
            contains the labels as first column then configuration columns
            then observation columns
        debug: boolean, default=False
        centroid_strategy: str ('all' or 'shared')
            'all' means compute the centroids from all given configurations
            used for training the autoencoder
            'shared' means compute the centroids from X_shared
        X_shared: numpy array, default None
            Only meaningful if centroid_strategy is 'shared'.
            Same format as X
        log_time: boolean, default=False
            Whether or not to log the time to train the autoencoder.
        refit_attempt: int
            How many times we're attempting to refit again the autoencoder
            because of vanishing centroids...

        """
        self.refit_attempt = refit_attempt
        if refit_attempt > 0:
            logging.warn("Refitting autoencoder (attempt #: {})".format(
                refit_attempt))
        t0 = time.time()
        labels = X[:, 0]
        configurations = X[:, 1:1 + len(self.knob_cols)]
        Y = X[:, 1 + len(self.knob_cols):]
        YY = Y.copy()

        early_stopping = self.early_stopping
        patience = self.patience
        validation_set = self.validation_set

        if early_stopping:
            config_train, config_val, Y_train, Y_val, labels_train, labels_val, _ = train_test_split_(
                configurations, Y, labels, test_size=.1, shuffle=False)
        else:
            config_train = configurations
            Y_train = Y
            labels_train = labels
        encoding_vector_index = int(len(self.hidden_layer_sizes) / 2)

        if not self._fitted:
            self.dim_Y = np.shape(Y_train)[1]
            self._architecture = self.__build_nn_architecture(
                self.dim_Y, self.hidden_layer_sizes)
            self._placeholders, self._weights, self._biases, self._outputs = self._architecture
            self._saver = Saver(
                self._weights[:encoding_vector_index + 1] + self._biases)
            self._fitted = True

        out = self._outputs[-1]
        encoded_value = self._outputs[encoding_vector_index]
        config_approx = encoded_value[:, -len(self.knob_cols):]
        encoded_value = encoded_value[:, :-len(self.knob_cols)]
        config = self._placeholders[0]
        Y = self._placeholders[-1]

        mse_err = tf.reduce_mean(tf.square(Y - out))
        config_recons_mse = tf.reduce_mean(tf.square(config - config_approx))
        err = mse_err + self.lamda * config_recons_mse
        if self.lamda < 1e-9:
            err = mse_err
        elif self.lamda > 1e9:
            err = config_recons_mse

        train_step = tf.compat.v1.train.AdamOptimizer(
            self.initial_learning_rate).minimize(err)

        n_train = np.shape(Y_train)[0]
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.log = {}
        self.log['obs_val_val'] = []
        self.log['config_recons_val'] = []
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            min_err = np.inf
            best_epoch = -1
            count = 0
            n_epochs = self.n_iter
            exited_early_stopping = False
            if debug:
                first_term_initial = mse_err.eval(
                    feed_dict=self.__get_feed_dict(
                        config_train, Y_train, self._placeholders))
                second_term_initial = config_recons_mse.eval(
                    feed_dict=self.__get_feed_dict(
                        config_train, Y_train, self._placeholders))

            for e in range(n_epochs):
                if early_stopping and best_epoch > 0 and e > best_epoch + patience:
                    exited_early_stopping = True
                    break
                stop = 0
                config_shuf, Y_shuf = shuffle(
                    config_train, Y_train, random_state=self.random_state)
                n_train = len(Y_train)
                r1 = range(int(np.ceil(n_train / self.batch_size)))
                for i in r1:
                    Y_batch = Y_shuf[i *
                                     self.batch_size:(i + 1) * self.batch_size, :]
                    config_batch = config_shuf[i *
                                               self.batch_size:(i + 1) * self.batch_size, :]
                    train_step.run(feed_dict=self.__get_feed_dict(
                        config_batch, Y_batch, self._placeholders))

                loss = err.eval(feed_dict=self.__get_feed_dict(
                    config_train, Y_train, self._placeholders))
                self.history['loss'].append(loss)

                if debug:
                    mse_err_value = mse_err.eval(
                        feed_dict=self.__get_feed_dict(
                            config_train, Y_train, self._placeholders))
                    config_recons_err_value = config_recons_mse.eval(
                        feed_dict=self.__get_feed_dict(
                            config_train, Y_train, self._placeholders))

                if early_stopping:
                    val_loss = err.eval(
                        feed_dict=self.__get_feed_dict(
                            config_val, Y_val, self._placeholders))
                    self.history['val_loss'].append(val_loss)

                    if debug:
                        mse_err_value_ = mse_err.eval(
                            feed_dict=self.__get_feed_dict(
                                config_val, Y_val, self._placeholders))
                        config_recons_err_value_ = config_recons_mse.eval(
                            feed_dict=self.__get_feed_dict(
                                config_val, Y_val, self._placeholders))
                        self.log['obs_val_val'].append(mse_err_value_)
                        self.log['config_recons_val'].append(
                            config_recons_err_value_)

                    if val_loss < min_err:
                        self._saver.save_weights(sess)
                        min_err = val_loss
                        count += 1
                        best_epoch = e

            if early_stopping and exited_early_stopping:
                self._saver.restore_weights(sess)
            else:
                best_epoch = -1
                self._saver.save_weights(sess)

            if debug:
                first_term_final = mse_err.eval(
                    feed_dict=self.__get_feed_dict(
                        config_train, Y_train, self._placeholders))
                second_term_final = config_recons_mse.eval(
                    feed_dict=self.__get_feed_dict(
                        config_train, Y_train, self._placeholders))

                self.optim_log = [
                    first_term_initial, second_term_initial, first_term_final,
                    second_term_final]

            encoded_vals = encoded_value.eval(
                session=sess, feed_dict=self.__get_feed_dict(
                    configurations, YY, self._placeholders))

            if centroids_strategy == 'all':
                self._compute_centroids(encoded_vals, labels)
            elif centroids_strategy == 'shared':
                assert X_shared is not None
                labels_shared = X_shared[:, 0]
                configurations_shared = X_shared[:, 1:1 + len(self.knob_cols)]
                Y_shared = X_shared[:, 1 + len(self.knob_cols):]

                encoded_vals_shared = encoded_value.eval(
                    session=sess, feed_dict=self.__get_feed_dict(
                        configurations_shared, Y_shared, self._placeholders))
                self._compute_centroids(encoded_vals_shared, labels_shared)
            else:
                raise NotImplementedError

            count_zeros = 0
            for key in self.centroids:
                if np.sum(np.abs(self.centroids[key])) < 1e-10:
                    count_zeros += 1

            if count_zeros > 1 and self.auto_refit and(
                    refit_attempt + 1) < self.max_refit_attempts:
                self._fitted = False
                return self.fit(
                    X, debug=debug, centroids_strategy=centroids_strategy,
                    X_shared=X_shared, log_time=log_time,
                    refit_attempt=refit_attempt + 1)
            elif count_zeros > 1 and self.auto_refit:
                logging.warn("Attempted to refit the autoencoder {} times with vanishing centroids...".format(
                    self.max_refit_attempts))

            t_end = time.time()
            fitting_time = t_end - t0
            self._last_fit_duration = fitting_time

            if log_time:
                logging.info(
                    "[AE fitting time]: {} minutes and {} seconds".format(
                        fitting_time // 60,
                        int(fitting_time % 60)))
            if debug:
                return encoded_vals, best_epoch

            return encoded_vals

    def transform(self, X):
        configs = X[:, 1:1 + len(self.knob_cols)]
        if np.ndim(X) > 1:
            Y = X[:, 1 + len(self.knob_cols):]
        else:
            aux = np.shape(X[1 + len(self.knob_cols):])[0]
            Y = np.reshape(X[1 + len(self.knob_cols):], [1, aux])

        try:
            assert self._fitted == True
            encoding_vector_index = int(len(self.hidden_layer_sizes) / 2)

            out = self._outputs[-1]
            encoded_value = self._outputs[encoding_vector_index][
                :, : -len(self.knob_cols)]

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                self._saver.restore_weights(sess)
                dico = self.__get_feed_dict(configs, Y, self._placeholders)
                encodings = encoded_value.eval(session=sess, feed_dict=dico)

        except AssertionError:
            print("Error: needs to call fit before transform can be invoked")
            raise
        return encodings

    def __create_activation_map(self):
        self._activ_func = {}
        self._activ_func['relu'] = tf.nn.relu
        self._activ_func['sigmoid'] = tf.nn.sigmoid
        self._activ_func['tanh'] = tf.nn.tanh
        self._activ_func[None] = identity_tensor
        self._activ_func[''] = identity_tensor
        self._activ_func['linear'] = identity_tensor

    def __get_feed_dict(self, config, Y, placeholders):
        config_pl = placeholders[0]
        Y_true = placeholders[-1]
        feed_dict = {config_pl: config, Y_true: Y}
        return feed_dict

    def __make_fc_layers(
            self, Y, dim_Y, hidden_dimensions, activations, trainable):
        """
        Creates the weights, biases and outputs of the autoencoder with
        tied weights option.
        """
        size = len(hidden_dimensions)
        encoding_vector_index = int(len(hidden_dimensions) / 2)
        dimensions = hidden_dimensions + [dim_Y]
        activations = activations.copy()
        for i in range(len(activations)):
            activations[i] = self._activ_func[activations[i]]

        weights = []
        biases = []
        outputs = []
        for i in range(len(dimensions)):
            if i == 0:
                r = 4 * np.sqrt(6 / (dim_Y + dimensions[i]))
                Wi = weight_variable(
                    [dim_Y, dimensions[i]],
                    trainable=trainable[i],
                    init_std=0.1)
                bi = bias_variable(
                    [dimensions[i]], trainable=trainable[i], init=0.1)
                oi = tf.matmul(Y, Wi) + bi

            elif i <= encoding_vector_index:
                r = 4 * np.sqrt(6 / (dimensions[i - 1] + dimensions[i]))
                Wi = weight_variable(
                    [dimensions[i - 1],
                     dimensions[i]],
                    trainable=trainable[i],
                    init_std=0.1)
                bi = bias_variable(
                    [dimensions[i]], trainable=trainable[i], init=0.1)
                oi = tf.matmul(outputs[-1], Wi) + bi
                oi = activations[i](oi)
            elif i > encoding_vector_index:
                Wi = tf.transpose(weights[size - i])  # tied weights
                bi = bias_variable(
                    [dimensions[i]], trainable=trainable[i], init=0.1)
                oi = tf.matmul(outputs[-1], Wi) + bi
                oi = activations[i](oi)
            weights.append(Wi)
            biases.append(bi)
            outputs.append(oi)

        return weights, biases, outputs

    def __build_nn_architecture(self, dim_Y, hidden_dimensions):
        """
        Builds the autoencoder architecture given topology description
        """
        tf.compat.v1.disable_eager_execution()
        Y = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_Y])
        config = tf.compat.v1.placeholder(
            tf.float32, shape=[None, len(self.knob_cols)])
        placeholders = [config, Y]
        weights, biases, outputs = self.__make_fc_layers(
            Y, dim_Y, hidden_dimensions, self.activations, trainable=[True] *
            (len(hidden_dimensions) + 1))
        architecture = [placeholders, weights, biases, outputs]
        return architecture

    def get_encodings(self, labels):
        """
        Returns encodings given a set of labels (aliases or job ids) after
        calculating the centroids computed for each job (by averaging over
        its job traces encodings)
        """
        if self.centroids is not None:
            encodings = list(
                map(lambda x: self.centroids[int(x)], list(labels)))
            encodings = np.asarray(encodings)
            return encodings
        return None

    def get_reconstruction(self, X, y=None):
        if np.ndim(X) > 1:
            Y = X[:, 1 + len(self.knob_cols):]
        else:
            aux = np.shape(X[1 + len(self.knob_cols):])[0]
            Y = np.reshape(X[1 + len(self.knob_cols):], [1, aux])
        n = len(Y)
        fake_config = np.zeros([n, len(self.knob_cols)])

        try:
            assert self._fitted == True
            out = self._outputs[-1]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self._saver.restore_weights(sess)
                dico = self.__get_feed_dict(fake_config, Y, self._placeholders)
                reconstruction = out.eval(session=sess, feed_dict=dico)
                return reconstruction

        except AssertionError:
            print("Error: needs to call fit before transform can be invoked")
            raise

    def serialize(self, filepath):
        recons_info = self.get_persist_info()
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        np.save(filepath, recons_info)

    def persist(self, filepath):
        self.serialize(filepath)

    def get_persist_info(self):
        recons_info = {'best_params': self._saver.best_params,
                       'hidden_layer_sizes': self.hidden_layer_sizes,
                       'dim_Y': self.dim_Y, 'centroids': self.centroids,
                       '_last_fit_duration': self._last_fit_duration,
                       'altered_centroids': self.altered_centroids,
                       'refit_attempt': self.refit_attempt,
                       'auto_refit': self.auto_refit,
                       'max_refit_attempts': self.max_refit_attempts}
        return recons_info

    def load(self, filepath):
        recons_info = np.load(filepath, allow_pickle=True)[()]
        nn_weights = recons_info['best_params']
        self.hidden_layer_sizes = recons_info['hidden_layer_sizes']
        self.dim_Y = recons_info['dim_Y']
        self.centroids = recons_info['centroids']
        self.altered_centroids = recons_info['altered_centroids']
        self._last_fit_duration = recons_info['_last_fit_duration']
        self.max_refit_attempts = recons_info['max_refit_attempts']
        self.fake_fit(nn_weights)

    def load_(self, recons_info):
        nn_weights = recons_info['best_params']
        self.hidden_layer_sizes = recons_info['hidden_layer_sizes']
        self.dim_Y = recons_info['dim_Y']
        self.centroids = recons_info['centroids']
        self.altered_centroids = recons_info['altered_centroids']
        self._last_fit_duration = recons_info['_last_fit_duration']
        self.max_refit_attempts = recons_info['max_refit_attempts']
        self.auto_refit = recons_info['auto_refit']
        self.fake_fit(nn_weights)

    @staticmethod
    def build(
            n_iter=500, encoding_dim=3, depth=2, nh=20, activation='linear',
            initial_learning_rate=1e-3, solver='Adam', batch_size=32,
            random_state=10, early_stopping=False, patience=10, lamda=1e-1,
            knob_cols=None, auto_refit=True, max_refit_attempts=10):
        """
        Provides another interface (other than the constructor) for
        constructing autoencoder objects...

        """
        assert knob_cols is not None

        encoder_hidden_layers = [int(nh / (2**i)) for i in range(depth - 1)]
        if len(encoder_hidden_layers) > 0:
            if 0 in encoder_hidden_layers or encoder_hidden_layers[-1] < encoding_dim:
                return None
        decoder_hidden_layers = encoder_hidden_layers[::-1]
        hidden_layer_sizes = encoder_hidden_layers + \
            [encoding_dim] + decoder_hidden_layers
        activations = [activation] * 2 * depth
        ae = FancyAutoEncoder(
            n_iter, hidden_layer_sizes, activations, initial_learning_rate,
            solver=solver, batch_size=batch_size, random_state=random_state,
            early_stopping=early_stopping, patience=patience, lamda=lamda,
            knob_cols=knob_cols, auto_refit=auto_refit,
            max_refit_attempts=max_refit_attempts)
        return ae

    @staticmethod
    def valid_params(ae_params, encoding_size, n_knob_cols):
        nh = ae_params['nh']
        depth = ae_params['depth']
        if depth >= 2:
            return (nh / (2**(depth - 2))) > n_knob_cols + encoding_size
        return True
