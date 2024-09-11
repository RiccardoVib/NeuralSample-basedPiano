import tensorflow as tf


class FiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, trainable=False, **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias, trainable=trainable)
        self.glu = GLU(in_size=self.in_size, trainable=trainable)

    def call(self, x, c):
        c = self.dense(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)

        return x


class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, trainable=False, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias, trainable=trainable)

    def call(self, x):
        x = self.dense(x)
        out, gate = tf.split(x, 2, axis=self.dim)
        #gate = tf.keras.activations.softsign(gate)
        #gate = tf.keras.activations.sigmoid(gate)
        gate = tf.keras.activations.swish(gate)
        x = tf.multiply(out, gate)
        return x
