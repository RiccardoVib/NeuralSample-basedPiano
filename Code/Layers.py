import tensorflow as tf
import math as m
import numpy as np
from ConditioningLayer import FiLM, GLU
from Mamba import ResidualBlock
import math

class MambaLay(tf.keras.layers.Layer):
    def __init__(self, units, projection_expand_factor=2, model_input_dims=2, trainable=False, type=tf.float32):
        super(MambaLay, self).__init__()
        layer_id = np.round(np.random.randint(0, 1000), 4)
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16
        model_states = units//2
        conv_use_bias, dense_use_bias = True, True
        conv_kernel_size = 4
        self.block = ResidualBlock(layer_id, model_input_dims, self.model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size,
                      self.delta_t_rank, model_states, name=f"Residual_{0}", trainable=trainable)
        #self.dense = tf.keras.layers.Dense(units, activation=tf.nn.gelu, trainable=trainable)
        self.dense = tf.keras.layers.Dense(model_states, activation=tf.nn.gelu, trainable=trainable)

    def call(self, x):
        x = self.block(x)
        x = self.dense(x)
        return x

class EnhancementLayerLSTM(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, units, bias=True, dim=-1, trainable=True, type=tf.float32):
        super(EnhancementLayerLSTM, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        #self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps), trainable=trainable)
        self.lstm = tf.keras.layers.LSTM(units, stateful=True, trainable=trainable)
        self.film = FiLM(units, trainable=trainable)
        self.out = tf.keras.layers.Dense(steps, trainable=trainable)

    def reset(self):
        print("")

    def call(self, x, c):
        #x = self.proj(x)
        x = self.lstm(x)
        x = self.film(x, c)
        x = self.out(x)
        return x

class EnhancementLayerMamba(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, units, bias=True, dim=-1, trainable=True, type=tf.float32):
        super(EnhancementLayerMamba, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        #self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps), trainable=trainable)
        self.mamba = MambaLay(units, model_input_dims=steps,  trainable=trainable)
        self.film = FiLM(units//2, trainable=trainable)
        self.out = tf.keras.layers.Dense(steps, trainable=trainable)
    def reset(self):
        print("")
    def call(self, x, c):
        #x = self.proj(x)
        x = self.mamba(x)
        x = tf.squeeze(x, axis=1)
        x = self.film(x, c)
        x = self.out(x)
        return x

