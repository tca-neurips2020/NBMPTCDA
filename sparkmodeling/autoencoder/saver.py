import tensorflow as tf
import numpy as np


class Saver:
    def __init__(self, params):
        self.params = params
        self.best_params = []

    def save_weights(self, sess):
        self.best_params = []
        for param in self.params:
            self.best_params.append(param.eval(sess))

    def restore_weights(self, sess):
        if len(self.best_params) > 0:
            i = 0
            for param in self.params:
                sess.run(param.assign(self.best_params[i]))
                i += 1
        else:
            print("[Logger: No best parameters found]")

    def get_weights(self):
        return self.best_params

    def save_to_disk(self, filenames):
        i = 0
        for param in self.best_params:
            np.save(filenames[i], param)
            i += 1

    def load_from_disk(self, filenames):
        self.best_params = []
        for filename in filenames:
            param = np.load(filename)
            self.best_params.append(param)
