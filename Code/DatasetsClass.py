import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence


class DataGeneratorPickles(Sequence):

    def __init__(self, filename, data_dir, cond_dim, model, mini_batch_size=2048, batch_size=8, stateful=False,
                 type=np.float64):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """
        data = open(os.path.normpath('/'.join([data_dir, filename + '.pickle'])), 'rb')
        Z = pickle.load(data)
        z = Z['z']
        y = Z['y']
        # x_min = Z['x_min']
        x_max = Z['x_max']

        self.filename = filename
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.cond_dim = cond_dim
        self.model = model
        self.y = np.array(y, dtype=type)
        self.x = np.array(x_max, dtype=type)
        self.ratio = y.shape[1] // (mini_batch_size)
        self.stateful = stateful
        self.lim = self.ratio * self.mini_batch_size

        #########

        self.y = self.y[:, :self.lim]
        self.x = self.x[:, :self.lim]

        self.idj = 0
        self.idx = -1

        self.max_1 = (self.x.shape[1] // self.mini_batch_size) - 1
        self.max_2 = (self.x.shape[0] // self.batch_size)
        self.max = self.max_1 * self.max_2
        self.training_steps = self.max

        self.z = np.repeat(z[:, np.newaxis, :], self.y.shape[1], axis=1)

        self.prev_v = None
        self.prev_k = None
        self.model = model
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.z.shape[1])
        self.indices2 = np.arange(0, self.z.shape[0])
        self.idj = 0
        self.idx = -1
        self.model.reset_states()
        if self.stateful:
            self.model.layers[1].reset_states()

    def __len__(self):
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        if idx == 0:
            self.idj = 0
            self.idx = -1

        if idx % self.max_1 - 1 == 0 and idx != 1:
            self.idj += 1
            self.idx = -1

        self.idx += 1

        # get the indices of the requested batch
        indices = self.indices[self.idx * self.mini_batch_size:(self.idx + 1) * self.mini_batch_size]
        indices2 = self.indices2[self.idj * self.batch_size:(self.idj + 1) * self.batch_size]

        if self.prev_k != self.z[indices2[0], indices[0], 0] and self.prev_v != self.z[indices2[0], indices[0], 1]:
            if self.stateful:
                self.model.reset_states()
                self.model.layers[1].reset_states()

        X = self.x[indices2]
        X = X[:, indices]
        Y = self.y[indices2]
        Y = Y[:, indices]
        Z = self.z[indices2]
        Z = Z[:, indices]

        self.prev_v = self.z[indices2[0], indices[0], 1]
        self.prev_k = self.z[indices2[0], indices[0], 0]

        return [X, Z], Y