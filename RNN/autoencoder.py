import os
import tensorflow as tf
from keras import Sequential, layers, Model
from PIL import Image
from tensorflow import keras

h_dim = ''


class Ae(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim),
        ])

        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784),
        ])

    def call(self, inputs, training=None, mask=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)

        return x_hat


model = Ae()
model.build(input_shape=(None, 784))
model.summary()