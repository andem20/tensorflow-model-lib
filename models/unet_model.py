import tensorflow as tf
import layers
import models


class UNet(tf.keras.Model):
    def __init__(self, num_classes: int, stack: list[int]):
        super().__init__()
        self.num_classes = num_classes
        self.stack = stack
        self.encoder = models.ConvEncoder(stack=stack[:-1])
        self.decoder = models.UpsampleDecoder(stack[:-1][::-1])
        self.bottom_layer = layers.ConvBlock(stack[-1])
        self.outputs = tf.keras.layers.Conv2D(self.num_classes, 3, padding="same", activation="softmax")

    def call(self, inputs):
        x, skips = self.encoder(inputs)
        x = self.bottom_layer(x)
        x = self.decoder(x, skips[::-1])
        return self.outputs(x)