import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Attention(keras.layers.Layer):
    def __init__(self, gamma=0.01, trainable=True):
        super().__init__(trainable=trainable)
        self._gamma = gamma
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None

    def build(self, input_shape):
        c = input_shape[-1]
        self.f = self.block(c//8)     # reduce channel size, reduce computation
        self.g = self.block(c//8)     # reduce channel size, reduce computation
        self.h = self.block(c//8)     # reduce channel size, reduce computation
        self.v = keras.layers.Conv2D(c, 1, 1)              # scale back to original channel size
        

    @staticmethod
    def block(c):
        return keras.Sequential([
            keras.layers.Conv2D(c, 1, 1),   # [n, w, h, c] 1*1conv
            keras.layers.Reshape((-1, c)),          # [n, w*h, c]
        ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        g = self.g(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        h = self.h(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        s = tf.matmul(f, g, transpose_b=True)   # [n, w*h, c//8] @ [n, c//8, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c//8] = [n, w*h, c//8]
        s = inputs.shape        # [n, w, h, c]
        cs = context_wh.shape   # [n, w*h, c//8]
        context = tf.reshape(context_wh, [-1, s[1], s[2], cs[-1]])    # [n, w, h, c//8]
        o = self.v(self.gamma * context) + inputs   # residual
        return o
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'gamma':self._gamma, 'f':self.f, 'g':self.g, 'h':self.h, 'v':self.v, 'attention':self.attention}