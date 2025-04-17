import tensorflow as tf
from ConditioningLayer import FiLM
from Mamba import MambaLay

def create_model(cond_dim, units, model_type, mini_batch_size=2048, b_size=8, stateful=False):
    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='input')
    cond = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, cond_dim), name='cond')

    if model_type == 'LSTM':
        out = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True)(inputs)
        out = FiLM(units)(out, cond)

    if model_type == 'S6':
        out = MambaLay(model_states=units*8, projection_expand_factor=2, model_input_dims=1, conv_kernel_size=1,
                       batch_size=b_size, stateful=stateful)(inputs)

        out = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(out)
        #cond_ = tf.keras.layers.Dense(units)(cond)
        out = FiLM(units)(out, cond)

    #if model_type == 'LSTM':
    #    out = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True)(out)
    #if model_type == 'S6':
    #    out = MambaLay(model_states=units*8, projection_expand_factor=2, model_input_dims=1, conv_kernel_size=1,
    #                   batch_size=b_size, stateful=stateful)(out)
    #    out = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(out)

    out = tf.keras.layers.Dense(1)(out)

    model = tf.keras.models.Model([inputs, cond], out)

    model.summary()
    return model