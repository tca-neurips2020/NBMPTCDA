import tensorflow as tf
from sparkmodeling.autoencoder.saver import Saver
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np
import logging
import time


def weight_variable(shape, trainable=True, init_std=0.1, v1_compat_mode=False):
    if v1_compat_mode:
        initial = tf.random.truncated_normal(shape, stddev=init_std)
    else:
        initial = tf.truncated_normal(shape, stddev=init_std)
    return tf.Variable(initial, trainable=trainable)


def bias_variable(shape, trainable=True, init=0.1):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, trainable=trainable)


class TAutoEncoder:
    def __init__(self, input_dim, layer_sizes, activations, alpha=0.01,
                 gamma=1e-1, learning_rate=0.001, batch_size=128, n_epochs=40,
                 early_stopping=True, patience=5, v1_compat_mode=False,
                 random_state=42):
        """
        Siamese Neural network with triplet-loss implemented and featuring
        a reconstruction term for (1) anchor, (2) positive (3) negative


        The layer_sizes is only defined for the encoder part and not
        the decoder part...

        gamma: coefficient that multiplies the sum of the 3 reconstruction terms
        alpha: margin used within the triplet loss...

        """
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        if v1_compat_mode:
            self.session = tf.compat.v1.Session()
        else:
            self.session = tf.Session()
        self.best_epoch_ = None
        self.last_fit_duration_ = None
        self.centroids = None
        self.altered_centroids = None
        self.gamma = gamma
        self.v1_compat_mode = v1_compat_mode

    def compile(self):
        if self.v1_compat_mode:
            tf.compat.v1.disable_eager_execution()
        self._init_architecture()
        anchor_encoding = self.forward_pass(self.placeholders['anchor'])
        pos_encoding = self.forward_pass(self.placeholders['pos'])
        neg_encoding = self.forward_pass(self.placeholders['neg'])

        ap_norm = tf.norm(
            tf.square(anchor_encoding - pos_encoding),
            keepdims=True, axis=1)
        an_norm = tf.norm(
            tf.square(anchor_encoding - neg_encoding),
            keepdims=True, axis=1)
        triplet_loss = tf.nn.relu(ap_norm - an_norm + self.alpha)
        triplet_loss = tf.reduce_mean(triplet_loss)

        anchor_output = self.full_forward_pass(self.placeholders['anchor'])
        pos_output = self.full_forward_pass(self.placeholders['pos'])
        neg_output = self.full_forward_pass(self.placeholders['neg'])
        anchor_recons_loss = tf.reduce_mean(
            tf.square(anchor_output - self.placeholders['anchor']))
        pos_recons_loss = tf.reduce_mean(
            tf.square(pos_output - self.placeholders['pos']))
        neg_recons_loss = tf.reduce_mean(
            tf.square(neg_output - self.placeholders['neg']))

        recons_loss = anchor_recons_loss + pos_recons_loss + neg_recons_loss

        loss = triplet_loss + self.gamma * recons_loss

        if self.v1_compat_mode:
            adam = tf.compat.v1.train.AdamOptimizer
        else:
            adam = tf.train.AdamOptimizer

        train_step = adam(self.learning_rate).minimize(loss)

        self.loss = loss
        self.train_step = train_step

    def forward_pass(self, input_pl):
        output = input_pl
        for i in range(len(self.weights)):
            output = tf.matmul(output, self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                output = tf.nn.relu(output)
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
        logging.debug("Full forward pass: ")
        for i in range(len(self.decoder_weights)):
            logging.debug("{}: prev_output_shape: {} \t weights shape: {} \t bias shape: {}".format(
                i, output.shape, self.decoder_weights[i].shape, self.decoder_biases[i].shape))
            output = tf.matmul(
                output, self.decoder_weights[i]) + self.decoder_biases[i]
            if self.activations[len(self.activations) - i - 1] == 'relu':
                output = tf.nn.relu(output)
        return output

    def _init_architecture(self):
        if self.v1_compat_mode:
            anchor_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self.input_dim))
            pos_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self.input_dim))
            neg_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self.input_dim))
        else:
            anchor_pl = tf.placeholder(tf.float32, shape=(None, self.input_dim))
            pos_pl = tf.placeholder(tf.float32, shape=(None, self.input_dim))
            neg_pl = tf.placeholder(tf.float32, shape=(None, self.input_dim))

        self.placeholders = {
            'anchor': anchor_pl,
            'pos': pos_pl,
            'neg': neg_pl
        }

        weights = []
        biases = []
        i_dim = self.input_dim
        for layer_size in self.layer_sizes:
            w = weight_variable(
                [i_dim, layer_size],
                v1_compat_mode=self.v1_compat_mode)
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

    def get_fd(self, X_a, X_p, X_n):
        return {self.placeholders['anchor']: X_a,
                self.placeholders['pos']: X_p,
                self.placeholders['neg']: X_n
                }

    def eval_var(self, var, X_a, X_p, X_n):
        return var.eval(
            feed_dict=self.get_fd(X_a, X_p, X_n),
            session=self.session)

    def fit_idxs(
            self, triplet_idxs, fetch_method, lods, log_time=False,
            verbose=False):
        t0 = time.time()

        triplet_idxs = np.array(triplet_idxs)

        if self.early_stopping:
            triplet_idxs, triplet_idxs_val = train_test_split(
                triplet_idxs, shuffle=False)
            self.history = {'loss': [], 'val_loss': []}
        else:
            self.history = {'loss': []}

        n_points = len(triplet_idxs)
        sess = self.session
        if self.v1_compat_mode:
            sess.run(tf.compat.v1.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        n_batches = int(np.ceil(n_points / self.batch_size))
        bs = self.batch_size

        best_epoch = -1
        min_err = np.inf

        for e in range(self.n_epochs):
            if self.early_stopping and best_epoch > 0 and e > best_epoch + self.patience:
                exited_early_stopping = True
                break

            triplet_idxs = shuffle(
                triplet_idxs, random_state=self.random_state + e)

            loss_value = []
            for i in range(n_batches):
                n_skipped = 0
                if (i % 1000) == 0:
                    if verbose:
                        logging.info(
                            "Epoch: {} \t step: {}/{} batches".format(e, i, n_batches))

                triplets_idxs_batch = triplet_idxs[i*bs:(i+1)*bs, :]
                xa_batch, xp_batch, xn_batch = fetch_method(
                    triplets_idxs_batch, lods)
                batch_loss_value = self.eval_var(
                    self.loss, xa_batch, xp_batch, xn_batch)
                if np.isfinite(batch_loss_value):
                    self.train_step.run(
                        feed_dict=self.get_fd(
                            xa_batch, xp_batch, xn_batch),
                        session=self.session)
                    loss_value.append(batch_loss_value)
                else:
                    n_skipped += 1
            loss_value = np.mean(loss_value)
            self.history['loss'].append(loss_value)
            if not np.isfinite(loss_value):
                logging.warn(
                    "Training stopped: nan or inf loss value")
                break
            if not self.early_stopping and verbose:
                logging.info("===> Epoch: {} \t loss: {:.6f}".format(
                    e, loss_value))
            else:
                n_val_batches = int(
                    np.ceil(len(triplet_idxs_val) / self.batch_size))
                val_loss_value = []
                for i in range(n_val_batches):
                    triplets_idxs_batch_val = triplet_idxs_val[i*bs:(i+1)*bs, :]
                    xa_batch_val, xp_batch_val, xn_batch_val = fetch_method(
                        triplets_idxs_batch_val, lods)
                    batch_loss_value = self.eval_var(
                        self.loss, xa_batch_val, xp_batch_val, xn_batch_val)
                    if np.isfinite(batch_loss_value):
                        val_loss_value.append(batch_loss_value)
                val_loss_value = np.nanmean(val_loss_value)

                self.history['val_loss'].append(val_loss_value)
                if not np.isfinite(val_loss_value):
                    logging.warn(
                        "Training stopped: nan or inf validation loss value")
                    break
                if val_loss_value < min_err:
                    min_err = val_loss_value
                    best_epoch = e
                    self.best_epoch_ = e
                    self.saver.save_weights(self.session)
                    if verbose:
                        logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f} ** (new best epoch)".format(
                            e, loss_value, val_loss_value))
                else:
                    if verbose:
                        logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f}".format(
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
                "[Triplet fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))
        return self.history

    def fit(self, X_a, X_p, X_n, log_time=False):
        assert len(X_a) == len(X_p)
        assert len(X_p) == len(X_n)

        t0 = time.time()

        if self.early_stopping:
            X_a, X_a_val, X_p, X_p_val, X_n, X_n_val = train_test_split(
                X_a, X_p, X_n, shuffle=False)
            self.history = {'loss': [], 'val_loss': []}
        else:
            self.history = {'loss': []}

        n_points = len(X_a)
        sess = self.session
        if self.v1_compat_mode:
            sess.run(tf.compat.v1.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
        self.history['loss'].append(self.eval_var(self.loss, X_a, X_p, X_n))
        if self.early_stopping:
            self.history['val_loss'].append(self.eval_var(
                self.loss, X_a_val, X_p_val, X_n_val))
        n_batches = int(np.ceil(n_points / self.batch_size))
        bs = self.batch_size

        best_epoch = -1
        min_err = np.inf

        logging.info("Initial loss(es): {}".format(self.history))
        for e in range(self.n_epochs):
            if self.early_stopping and best_epoch > 0 and e > best_epoch + self.patience:
                exited_early_stopping = True
                break

            X_a, X_p, X_n = shuffle(
                X_a, X_p, X_n, random_state=self.random_state+e)
            for i in range(n_batches):
                if (i % 1000) == 0:
                    logging.info(
                        "Epoch: {} \t step: {}/{} batches".format(e, i, n_batches))
                xa_batch = X_a[i*bs:(i+1)*bs, :]
                xp_batch = X_p[i*bs:(i+1)*bs, :]
                xn_batch = X_n[i*bs:(i+1)*bs, :]
                self.train_step.run(feed_dict=self.get_fd(
                    xa_batch, xp_batch, xn_batch),
                    session=self.session)
            loss_value = self.eval_var(self.loss, X_a, X_p, X_n)
            self.history['loss'].append(loss_value)
            if not self.early_stopping:
                logging.info("===> Epoch: {} \t loss: {:.6f}".format(
                    e, loss_value))
            else:
                val_loss_value = self.eval_var(
                    self.loss, X_a_val, X_p_val, X_n_val)
                self.history['val_loss'].append(val_loss_value)
                if val_loss_value < min_err:
                    min_err = val_loss_value
                    best_epoch = e
                    self.best_epoch_ = e
                    self.saver.save_weights(self.session)
                    logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f} ** (new best epoch)".format(
                        e, loss_value, val_loss_value))
                else:
                    logging.info("===> Epoch: {} \t loss: {:.6f} \t val_loss: {:.6f}".format(
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
                "[TripletAutoEncoder fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))
        return self.history

    def transform(self, X):
        output_var = self.forward_pass(self.placeholders['anchor'])
        output = output_var.eval(feed_dict={
            self.placeholders['anchor']: X
        }, session=self.session)
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
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'gamma': self.gamma,
            'v1_compat_mode': self.v1_compat_mode,
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
        return TAutoEncoder.make_instance(data['signature'], data['other'])

    @staticmethod
    def make_instance(signature_data, other_data):
        instance = TAutoEncoder(**signature_data)
        instance.compile()
        instance.saver.best_params = other_data['best_weights'].copy()
        instance.saver.restore_weights(instance.session)
        instance.history = other_data['history'].copy()
        instance.last_fit_duration_ = other_data['last_fit_duration']
        instance.best_epoch_ = other_data['best_epoch']
        if 'centroids' in other_data:
            instance.centroids = other_data['centroids']
            instance.altered_centroids = other_data['altered_centroids']
        return instance

    @staticmethod
    def load_from_file(fpath):
        data = np.load(fpath, allow_pickle=True)[()]
        return TAutoEncoder.make_instance(data['signature'],
                                          data['other'])
