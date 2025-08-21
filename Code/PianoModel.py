# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "Neural Sampled-based Piano Synthesis" in proceedings of the 25th Digital Audio Effect Conference, Ancona, Italy.

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