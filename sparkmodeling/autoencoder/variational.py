# Original paper (Kingma et al.):
# https://arxiv.org/pdf/1312.6114.pdf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten
from scipy.stats import norm
import time
import logging
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.get_logger().setLevel('INFO')


def sampling_func(inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    latent_dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim),
                               mean=0., stddev=1.)
    return z_mean + tf.exp(z_log_var / 2) * epsilon


class VAE:
    """
    Implementation of the Beta variational auto-encoder
    """

    def __init__(self, input_dim, layer_sizes, beta=1,
                 recons_type="xent", intermediate_activation='relu',
                 learning_rate=0.001, batch_size=128, n_epochs=100,
                 early_stopping=True, patience=10, random_state=42):
        """
        Implements a contractive auto-encoder

        input_dim: int
            Dimension of the input vector
        layer_sizes: list
            List of integers concerning the size of the layers in the encoder
            part.
        beta: float
            Coefficient that multiplies the KL term of the loss function
        recons_type: str
            Type of the reconstruction: can be 'xent' or 'mse'
        learning_rate: float
            Learning rate used for updating weights when minimizing the loss
            function.
        batch_size: int
            Size of the batch on which one training step is done
        intermediate_activation: str, Default: 'relu'
            Activation used in intermediate layers. 
        n_epochs: int
            Number of epochs for training
        early_stopping: boolean
            Whether to use early stopping mechanism to stop training and avoid
            overfitting.
        patience: int
            Number of epochs to watch before stopping the training. Only
            useful if early_stopping is True.
        random_state: int
            Seed to be used with pseudo-random number generator.

        """
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.best_epoch_ = None
        self.last_fit_duration_ = None
        self.beta = beta
        self.inter_activation = intermediate_activation
        self.recons_type = recons_type
        self.centroids = None
        self.altered_centroids = None

    def summary(self):
        return self.model.summary()

    def make_encoder(self):
        x = Input(shape=(self.input_dim,))
        hidden = x
        for i in range(len(self.layer_sizes) - 1):
            hidden = Dense(self.layer_sizes[i],
                           activation=self.inter_activation)(hidden)
        latent_dim = self.layer_sizes[-1]
        z_mean = Dense(latent_dim)(hidden)
        z_log_var = Dense(latent_dim)(hidden)
        return Model(inputs=x, outputs=[z_mean, z_log_var],
                     name="fc_encoder")

    def make_decoder(self):
        latent_dim = self.layer_sizes[-1]
        decoder_input = Input(shape=(latent_dim,))
        x = decoder_input
        for i in range(len(self.layer_sizes) - 1)[::-1]:
            x = Dense(
                self.layer_sizes[i],
                activation=self.inter_activation)(x)
        x = Dense(self.input_dim, activation='sigmoid')(x)
        return Model(inputs=decoder_input, outputs=x, name="fc_decoder")

    def compile(self):
        x = Input(shape=(self.input_dim, ), name="input")
        latent_dim = self.layer_sizes[-1]
        encoder = self.make_encoder()
        decoder = self.make_decoder()
        sampling_layer = Lambda(sampling_func, output_shape=(latent_dim,),
                                name="sampling_layer")

        z_mean, z_log_var = encoder(x)
        z = sampling_layer([z_mean, z_log_var])
        x_decoded_mean = decoder(z)
        model = Model(inputs=x, outputs=x_decoded_mean)

        if self.recons_type == "xent":
            recons_loss = self.input_dim * metrics.binary_crossentropy(
                Flatten()(x), Flatten()(x_decoded_mean))
        elif self.recons_type == "mse":
            recons_loss = metrics.mse(x, x_decoded_mean)

        kl_loss = - 0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        vae_loss = tf.reduce_mean(recons_loss + self.beta * kl_loss)

        model.add_loss(vae_loss)
        adam = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam)
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, X, n_epochs=None, verbose=1, log_time=False):
        if n_epochs is not None:
            self.n_epochs = n_epochs
        t0 = time.time()
        callbacks = None
        if self.early_stopping:
            X, X_val = train_test_split(X, shuffle=False)
            es_cb = EarlyStopping(monitor='val_loss', patience=self.patience)
            callbacks = [es_cb]
        history = self.model.fit(
            X, epochs=self.n_epochs, batch_size=self.batch_size,
            validation_data=(X_val, None),
            callbacks=callbacks, verbose=verbose)

        self.trained_weights = self.model.get_weights()
        tend = time.time()
        fitting_time = tend - t0
        self.last_fit_duration_ = fitting_time
        self.history_ = history.history

        if log_time:
            logging.info(
                "[VAE fitting time]: {} minutes and {} seconds".format(
                    fitting_time // 60,
                    int(fitting_time % 60)))

    def transform(self, X):
        latent_means, latent_log_vars = self.encoder(X)
        return latent_means, latent_log_vars

    def clone(self):
        """
        Returns a cloned version of the variational autoencoder
        """
        copy = VAE(self.input_dim, self.layer_sizes)
        for key, value in self.__dict__.items():
            if key != 'model' and key != 'encoder' and key != 'decoder':
                setattr(copy, key, value)
        copy.compile()
        copy.model.set_weights(copy.trained_weights)
        return copy

    def persist(self, filepath):
        """
        Serialize the current variational autoencoder to disk

        filepath: str
            filepath containing the filename with the npy extension. If
            no extension is provided, then automatically the file will
            have npy extension.
        """
        params = self.get_persist_info()
        np.save(filepath, params)

    def get_persist_info(self):
        params = {}
        for key, value in self.__dict__.items():
            if key != 'model' and key != 'encoder' and key != 'decoder':
                params[key] = value
        return params

    @staticmethod
    def load_from_file(fpath):
        params = np.load(fpath, allow_pickle=True)[()]
        instance = VAE(1, [])
        for key, value in params.items():
            setattr(instance, key, value)
        instance.compile()
        instance.model.set_weights(instance.trained_weights)
        return instance

    @staticmethod
    def build(input_dim=561, recons_type="xent",
              encoding_dim=5, depth=2, nh=20, intermediate_activation='relu',
              learning_rate=1e-3, batch_size=128, n_epochs=500,
              random_state=10, early_stopping=True, patience=10,
              beta=1):
        """
        Provides another interface (other than the constructor) for
        constructing autoencoder objects...
        """
        encoder_hidden_layers = [int(nh / (2**i)) for i in range(depth - 1)]
        if len(encoder_hidden_layers) > 0:
            if 0 in encoder_hidden_layers or encoder_hidden_layers[-1] < encoding_dim:
                return None
        hidden_layer_sizes = encoder_hidden_layers + [encoding_dim]

        ae = VAE(input_dim, hidden_layer_sizes, beta=beta,
                 recons_type=recons_type,
                 intermediate_activation=intermediate_activation,
                 learning_rate=learning_rate, batch_size=batch_size,
                 n_epochs=n_epochs, early_stopping=early_stopping,
                 patience=patience, random_state=random_state)
        return ae

    @staticmethod
    def valid_params(ae_params, encoding_size):
        nh = ae_params['nh']
        depth = ae_params['depth']
        if depth >= 2:
            return (nh / (2**(depth - 2))) > encoding_size
        return True
