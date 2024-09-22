import pickle
import os
from Utils import AttTime
import librosa
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, filename, data_dir, set, steps, model, batch_size=2800, type=np.float64):
        """
        Initializes a data generator object
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param set: which type of set
          :param steps: the number of timesteps per iteration
          :param steps: the neural model
          :param batch_size: The size of each batch returned by __getitem__
        """
        data = open(os.path.normpath('/'.join([data_dir, filename + '.pickle'])), 'rb')
        Z = pickle.load(data)
        y, keys, velocities = Z[set]
        if set == 'train':

            #lower
            # x = y[6:7] #lower 6:7, max 8:9, mid 7:8
            # y = np.delete(y, 6, axis=0) #8,7
            # velocities = np.delete(velocities, 6, axis=0) #8,7

            #max
            x = y[8:9] #lower 6:7, max 8:9, mid 7:8
            y = np.delete(y, 8, axis=0) #8,7
            velocities = np.delete(velocities, 8, axis=0) #8,7
            y_v, _, velocities_v = Z['val']
            y = np.concatenate([y, y_v], axis=0)
            velocities = np.concatenate([velocities, velocities_v], axis=0)

        if set == 'val':
            xy, _, velocities_x = Z['train']
            #lower
            # x = x[6:7]
            # max
            x = xy[8:9]
            y = xy[5:6]
            velocities = velocities_x[5:6]

        self.filename = filename
        self.batch_size = batch_size
        self.steps = steps
        self.y = np.array(y, dtype=type)
        self.x = np.array(x, dtype=type)
        self.ratio = y.shape[1] // (steps)

        self.lim = self.ratio//self.batch_size*self.batch_size
        self.ratio = self.lim // (steps)

        ###metadata
        self.velocities = velocities.reshape(-1, 1)/111
        self.n_note = self.velocities.shape[0]

        #########

        self.y = self.y[:, :self.lim].reshape(-1, steps)
        self.x = np.repeat(x, self.n_note, axis=0)
        self.x = self.x[:, :self.lim].reshape(-1, steps)

        self.velocities = np.repeat(self.velocities, self.ratio, axis=0).reshape(-1, 1)

        self.prev_v = None
        self.model = model
        self.on_epoch_end()

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.velocities.shape[0])

    def __len__(self):
        # compute the needed number of iteration before conclude one epoch
        return int(self.velocities.shape[0]/self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # reset states if processing new velocity
        if self.prev_v != self.velocities[indices[0], 0]:
            self.model.reset_states()
            self.model.layers[2].reset()

        self.prev_v = self.velocities[indices[0], 0]


        inputs = [self.x[indices].reshape(self.batch_size, self.steps), self.velocities[indices]]
        targets = self.y[indices].reshape(self.batch_size, self.steps)

        return (inputs, (targets))
