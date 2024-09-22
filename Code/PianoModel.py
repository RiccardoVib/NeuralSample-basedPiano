import tensorflow as tf
from Layers import EnhancementLayerLSTM, EnhancementLayerMamba

def create_model(cond_dim, input_dim, units, model_type, b_size=2399):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(b_size, 1, input_dim), name='input')
    cond_inputs = tf.keras.layers.Input(batch_shape=(b_size, cond_dim), name='cond')

    if model_type == 'LSTM':
        decoder_outputs = EnhancementLayerLSTM(b_size, 1, units//2)(inputs, cond_inputs)
    elif model_type == 'S6':
        decoder_outputs = EnhancementLayerMamba(b_size, 1, 26)(inputs, cond_inputs)

    model = tf.keras.models.Model([inputs, cond_inputs], decoder_outputs)

    model.summary()
    return model
