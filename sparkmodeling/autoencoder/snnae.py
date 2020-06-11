import tensorflow as tf
from sparkmodeling.autoencoder.saver import Saver
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np
import logging
import time
from copy import deepcopy


def weight_variable(shape, trainable=True, init_std=0.1):
    # tensorflow 2.0; 1.x counterpart is tf.truncated_normal
    initial = tf.random.truncated_normal(
        shape, stddev=init_std, dtype=tf.float64)
    return tf.Variable(initial, trainable=trainable, dtype=tf.float64)


def bias_variable(shape, trainable=True, init=0.1):
    initial = tf.constant(init, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, trainable=trainable, dtype=tf.float64)


class SNNAE:
    def __init__(self, input_dim, layer_sizes, activations, lamda=1e-1,
                 T=2, learning_rate=0.001, batch_size=128, n_epochs=100,
                 early_stopping=True, patience=10, random_state=42,
                 epsilon=1e-8):
        """
        Implements an auto-encoder with SNN loss

        layer_sizes: list
            It concerns only the encoder part and not the decoder part.
        lamda: term that multiplies the SNN term of the loss function.
        T: float, default=2
            Temperature term in the SNN loss

        epsilon: float, default=1e-8
            An Epsilon float number used for numerical stability
            (added to the log)

        """
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.session = tf.compat.v1.Session()
        self.best_epoch_ = None
        self.last_fit_duration_ = None
        self.centroids = None
        self.altered_centroids = None
        self.lamda = lamda
        self.epsilon = epsilon
        self.T = T

        assert len(activations) == len(layer_sizes)

    def get_log_at_i(self, x_batch, y_batch, i, selector):
        T = self.T
        b = self.batch_size
        ind_denom = selector
        denom = tf.reduce_sum(
            ind_denom * tf.exp(-tf.square(tf.norm(x_batch[i] - x_batch, axis=1))/T))

        ind_num = tf.reshape(selector, [b, 1])
        indic_same_label = tf.cast(
            tf.equal(y_batch[i],
                     y_batch),
            dtype=tf.float64)
        indic = tf.squeeze(tf.transpose(ind_num * indic_same_label))

        nume = tf.reduce_sum(
            indic * tf.exp(-tf.square(tf.norm(x_batch[i] - x_batch, axis=1))/T))

        return tf.math.log(self.epsilon + nume/denom)

    def get_snn_loss(self, encoding_var, label_pl):
        elems = np.arange(self.batch_size)
        selectors = np.ones([self.batch_size, self.batch_size])
        for i in range(self.batch_size):
            selectors[i, i] = 0

        elems = np.arange(self.batch_size)
        logs_batch = tf.map_fn(
            lambda t: self.get_log_at_i(
                encoding_var, label_pl, t[0], t[1]),
            (elems, selectors),
            dtype=tf.float64)
        snn = -tf.reduce_mean(logs_batch)

        return snn

    def compile(self):
        self._init_architecture()
        obs_approx = self.full_forward_pass(self.input_pl)
        encodings = self.forward_pass(self.input_pl)
        labels = self.label_pl

        recons_loss = tf.reduce_mean(tf.square(obs_approx - self.input_pl))
        snn_loss = self.get_snn_loss(encodings, labels)

        if self.lamda < 1e-9:
            loss = recons_loss
        else:
            loss = recons_loss + self.lamda * snn_loss

        train_step = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(loss)

        self.loss = loss
        self.recons_loss = recons_loss
        self.snn_loss = snn_loss
        self.train_step = train_step

    def forward_pass(self, input_pl):
        output = input_pl
        for i in range(len(self.weights)):
            output = tf.matmul(output, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                output = tf.nn.relu(output)
            elif self.activations[i] == 'sigmoid':
                output = tf.nn.sigmoid(output)
            elif self.activations[i] == '' or self.activations[i] is None:
                pass
            else:
                raise NotImplementedError(
                    "This activation ({}) is not yet implemented.".format(
                        self.activations[i]))
        return output

    def full_forward_pass(self, input_pl):
        encoding = self.forward_pass(input_pl)
        output = encoding
        for i in range(len(self.decoder_weights)):
            output = tf.matmul(
                output, self.decoder_weights[i]) + self.decoder_biases[i]
            if self.activations[len(self.activations) - i - 1] == 'relu':
                output = tf.nn.relu(output)
        return output

    def _init_architecture(self):
        tf.compat.v1.disable_eager_execution()
        self.input_pl = tf.compat.v1.placeholder(
            tf.float64, shape=(None, self.input_dim))
        self.label_pl = tf.compat.v1.placeholder(
            tf.int32, shape=(None, 1))

        weights = []
        biases = []
        i_dim = self.input_dim
        for layer_size in self.layer_sizes:
            w = weight_variable([i_dim, layer_size])
            b = bias_variable([layer_size])
            i_dim = layer_size
            weights.append(w)
            biases.append(b)

        decoder_weights = []
        decoder_biases = []
        for w in weights[::-1]:
            decoder_weights.append(tf.transpose(w))
            decoder_biases.append(bias_variable([int(w.shape[0])]))

        self.weights = weights
        self.biases = biases
        self.decoder_weights = decoder_weights
        self.decoder_biases = decoder_biases

        self.saver = Saver(self.weights + self.biases + self.decoder_biases)

    def get_fd(self, X, y=None):
        if y is None:
            return {
                self.input_pl: X
            }
        else:
            return {
                self.input_pl: X,
                self.label_pl: y
            }

    def eval_var(self, var, X):
        return var.eval(
            feed_dict=self.get_fd(X),
            session=self.session)

    def eval_var2(self, var, X, y):
        return var.eval(
            feed_dict=self.get_fd(X, y),
            session=self.session)

    def eval_snn(self, X, y, batches):
        snn_loss = []
        for i in range(len(batches)):
            idxs = batches[i]
            x_batch = X[idxs, :]
            y_batch = y[idxs, :]
            batch_snn_loss = self.eval_var2(
                self.snn_loss, x_batch, y_batch)
            if np.isfinite(batch_snn_loss):
                snn_loss.append(batch_snn_loss)
        snn_loss = np.mean(snn_loss)
        return snn_loss

    def log_losses(
            self, X, y, batches, val=False, verbose=False, e=0):
        recons_loss = self.eval_var(self.recons_loss, X)
        snn_loss = self.eval_snn(
            X, y, batches)
        if self.lamda < 1e-9:
            loss = recons_loss
        else:
            loss = recons_loss + self.lamda * snn_loss

        if not val:
            self.history['loss'].append(loss)
            self.history['recons_loss'].append(recons_loss)
            self.history['snn_loss'].append(snn_loss)
        else:
            self.history['val_loss'].append(loss)
            self.history['val_recons_loss'].append(recons_loss)
            self.history['val_snn_loss'].append(snn_loss)

        if verbose:
            if val:
                prefix = "[VAL]"
            else:
                prefix = "[TRAIN]"
            logging.info("{} Epoch {} - Losses: recons: {:.5f} \t snn: {:.5f} \t total: {:.5f}".format(
                prefix, e,  recons_loss, snn_loss, loss))
        return loss

    def log_losses(
            self, X, y, batches, val=False, verbose=False, e=0):
        recons_loss = self.eval_var(self.recons_loss, X)
        snn_loss = self.eval_snn(
            X, y, batches)
        if self.lamda < 1e-9:
            loss = recons_loss
        else:
            loss = recons_loss + self.lamda * snn_loss

        if not val:
            self.history['loss'].append(loss)
            self.history['recons_loss'].append(recons_loss)
            self.history['snn_loss'].append(snn_loss)
        else:
            self.history['val_loss'].append(loss)
            self.history['val_recons_loss'].append(recons_loss)
            self.history['val_snn_loss'].append(snn_loss)

        if verbose:
            if val:
                prefix = "[VAL]"
            else:
                prefix = "[TRAIN]"
            logging.info("{} Epoch {} - Losses: recons: {:.5f} \t second_term: {:.5f} \t total: {:.5f}".format(
                prefix, e,  recons_loss, snn_loss, loss))
        return loss

    def get_batches(self, slices, n_batches):
        batches = []
        tmp = deepcopy(slices)

        for i in range(n_batches):
            current_batch_idxs = []
            keys = list(tmp.keys())
            keys = shuffle(keys)

            for key in keys:
                if len(tmp[key]) < 4:
                    keys.remove(key)

            for key in keys:
                if len(tmp[key]) >= 4:
                    current_batch_idxs.extend(tmp[key][:4])
                    del tmp[key][:4]
                if len(current_batch_idxs) == self.batch_size:
                    break
            if len(current_batch_idxs) == self.batch_size:
                batches.append(current_batch_idxs)
        return batches

    def fit(self, X, y, log_time=False, verbose=False):
        t0 = time.time()

        self.history = {
            'loss': [],
            'recons_loss': [],
            'snn_loss': []}

        if self.early_stopping:
            repeat = True
            while repeat:
                X, X_val, y, y_val = train_test_split(X, y, shuffle=False)
                self.history['val_loss'] = []
                self.history['val_recons_loss'] = []
                self.history['val_snn_loss'] = []

                labels_to_rows = {}
                for i in range(len(y)):
                    if int(y[i]) in labels_to_rows:
                        labels_to_rows[int(y[i])].append(i)
                    else:
                        labels_to_rows[int(y[i])] = [i]

                labels_to_rows_val = {}
                for i in range(len(y_val)):
                    if int(y_val[i]) in labels_to_rows_val:
                        labels_to_rows_val[int(y_val[i])].append(i)
                    else:
                        labels_to_rows_val[int(y_val[i])] = [i]
                repeat = False

                for key in labels_to_rows:
                    if len(labels_to_rows[key]) < 2:
                        repeat = True
                        break
                for key in labels_to_rows_val:
                    if len(labels_to_rows_val[key]) < 2:
                        repeat = True
                        break
            n_batches_val = int(np.ceil(len(X_val) / self.batch_size)) - 1
        else:
            labels_to_rows = {}
            for i in range(len(y)):
                if y[i] in labels_to_rows:
                    labels_to_rows[y[i]].append(i)
                else:
                    labels_to_rows[y[i]] = [i]

        n_points = len(X)
        n_batches = int(np.ceil(n_points / self.batch_size)) - 1

        sess = self.session
        sess.run(tf.compat.v1.global_variables_initializer())

        batches = self.get_batches(labels_to_rows, n_batches)
        batches_val = self.get_batches(labels_to_rows_val, n_batches_val)

        loss_value = self.log_losses(
            X, y, batches, verbose=verbose)

        if not np.isfinite(loss_value):
            logging.warn(
                "Training not started because of non finite loss value: {}".format(loss_value))
            return None

        if self.early_stopping:
            val_loss_value = self.log_losses(
                X_val, y_val, batches_val, val=True, verbose=verbose)
            if not np.isfinite(val_loss_value):
                logging.warn("Training cancelled because of non finite val loss value: {}".format(
                    val_loss_value))
                return None

        best_epoch = -1
        min_err = np.inf

        logging.info("Initial loss(es): {}".format(self.history))
        for e in range(self.n_epochs):
            if self.early_stopping and best_epoch > 0 and e > best_epoch + self.patience:
                exited_early_stopping = True
                break

            batches = self.get_batches(labels_to_rows, n_batches)
            batches_val = self.get_batches(labels_to_rows_val, n_batches_val)

            for i in range(len(batches)):
                idxs = batches[i]
                x_batch = X[idxs, :]
                y_batch = y[idxs, :]
                self.train_step.run(feed_dict=self.get_fd(x_batch, y_batch),
                                    session=self.session)
            loss_value = self.log_losses(
                X, y, batches, verbose=verbose, e=e)
            if not np.isfinite(loss_value):
                logging.warn(
                    "Training stopped after {} epochs because of non finite loss value ({})".format(e, loss_value))
                break
            if self.early_stopping:
                val_loss_value = self.log_losses(
                    X_val, y_val, batches_val, val=True, verbose=verbose, e=e)
                if val_loss_value < min_err:
                    min_err = val_loss_value
                    best_epoch = e
                    self.best_epoch_ = e
                    self.saver.save_weights(self.session)
                    if verbose:
                        logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f} ** (new best epoch)".format(
                            e, loss_value, val_loss_value))

        if self.early_stopping:
            self.saver.restore_weights(self.session)
        else:
            self.saver.save_weights(self.session)

        tend = time.time()
        fitting_time = tend - t0
        self.last_fit_duration_ = fitting_time

        if log_time:
            logging.info(
                "[autoencoder fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))
        return self.history

    def transform(self, X):
        output_var = self.forward_pass(self.input_pl)
        output = output_var.eval(
            feed_dict={self.input_pl: X},
            session=self.session)
        return output

    def persist(self, fpath):
        data = self.get_persist_info()
        if os.path.dirname(fpath) != "":
            if not os.path.exists(os.path.dirname(fpath)):
                os.path.makedirs(os.path.dirname(fpath))
        np.save(fpath, data)

    def serialize(self, fpath):
        self.persist(fpath)

    def get_persist_info(self):
        signature_data = {
            'input_dim': self.input_dim,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'lamda': self.lamda,
            'T': self.T,
            'epsilon': self.epsilon
        }
        other_data = {
            'best_weights': self.saver.best_params,  # ws and bs
            'history': self.history,
            'best_epoch': self.best_epoch_,
            'last_fit_duration': self.last_fit_duration_,
            'centroids': self.centroids,
            'altered_centroids': self.altered_centroids
        }
        return {'signature': signature_data,
                'other': other_data}

    def clone(self):
        data = self.get_persist_info()
        return SNNAE.make_instance(data['signature'], data['other'])

    @staticmethod
    def make_instance(signature_data, other_data):
        instance = SNNAE(**signature_data)
        instance.compile()
        instance.saver.best_params = other_data['best_weights'].copy()
        instance.saver.restore_weights(instance.session)
        instance.history = other_data['history'].copy()
        instance.last_fit_duration_ = other_data['last_fit_duration']
        instance.best_epoch_ = other_data['best_epoch']

        return instance

    @staticmethod
    def load_from_file(fpath):
        data = np.load(fpath, allow_pickle=True)[()]
        return SNNAE.make_instance(data['signature'],
                                   data['other'])

    @staticmethod
    def build(input_dim=561, T=2,
              encoding_dim=5, depth=2, nh=20, activation='sigmoid',
              learning_rate=1e-3, batch_size=32, n_epochs=500,
              random_state=10, early_stopping=False, patience=10,
              lamda=1e-1, epsilon=1e-8):
        """
        Provides another interface (other than the constructor) for
        constructing autoencoder objects...
        """
        encoder_hidden_layers = [int(nh / (2**i)) for i in range(depth - 1)]
        if len(encoder_hidden_layers) > 0:
            if 0 in encoder_hidden_layers or encoder_hidden_layers[-1] < encoding_dim:
                return None
        hidden_layer_sizes = encoder_hidden_layers + [encoding_dim]
        activations = [activation] * depth

        ae = SNNAE(
            input_dim, hidden_layer_sizes, activations, lamda=lamda, T=T,
            learning_rate=learning_rate, batch_size=batch_size,
            n_epochs=n_epochs, early_stopping=early_stopping, patience=patience,
            random_state=random_state, epsilon=epsilon)
        return ae

    @staticmethod
    def valid_params(ae_params, encoding_size):
        if ae_params['patience'] >= ae_params['n_epochs']:
            return False
        nh = ae_params['nh']
        depth = ae_params['depth']
        if depth >= 2:
            return (nh / (2**(depth - 2))) > encoding_size
        return True
