import tensorflow as tf
import math as m
import numpy as np
from ConditioningLayer import FiLM, GLU
from Mamba import ResidualBlock
import math

class MambaLay(tf.keras.layers.Layer):
    def __init__(self, units, projection_expand_factor=2, model_input_dims=2, type=tf.float32):
        """
        Initializes the Mamba block
          :param units: the number of units
          :param projection_expand_factor: factor to which expand the input
          :param model_input_dims: the input size
        """
        super(MambaLay, self).__init__()
        layer_id = np.round(np.random.randint(0, 1000), 4)
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16
        model_states = units//2
        conv_use_bias, dense_use_bias = True, True
        conv_kernel_size = 4
        self.block = ResidualBlock(layer_id, model_input_dims, self.model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size,
                      self.delta_t_rank, model_states, name=f"Residual_{0}")
        self.dense = tf.keras.layers.Dense(model_states, activation=tf.nn.gelu)

    def call(self, x):
        x = self.block(x)
        x = self.dense(x)
        return x

class EnhancementLayerLSTM(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, units, type=tf.float32):
        """
        Initializes the LSTM-based network
          :param b_size: the batch size
          :param steps: the number of timesteps per iteration
          :param units: the number of size
        """
        super(EnhancementLayerLSTM, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        self.lstm = tf.keras.layers.LSTM(units, stateful=True)
        self.film = FiLM(units)
        self.out = tf.keras.layers.Dense(steps)

    def call(self, x, c):
        x = self.lstm(x)
        x = self.film(x, c)
        x = self.out(x)
        return x

class EnhancementLayerMamba(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, units, type=tf.float32):
        """
        Initializes the Mamba-based network
          :param b_size: the batch size
          :param steps: the number of timesteps per iteration
          :param units: the number of size
        """
        super(EnhancementLayerMamba, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.type = type
        self.b_size = b_size

        #self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps))
        self.mamba = MambaLay(units, model_input_dims=steps)
        self.film = FiLM(units//2)
        self.out = tf.keras.layers.Dense(steps)

    def call(self, x, c):
        x = self.mamba(x)
        x = tf.squeeze(x, axis=1)
        x = self.film(x, c)
        x = self.out(x)
        return x

