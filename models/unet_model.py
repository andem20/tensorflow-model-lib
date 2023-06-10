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


# def UNet(num_classes, input_shape, stack=[32, 64, 128, 256, 512]):   
#     inputs = tf.keras.Input(shape=input_shape)
#     x = inputs

#     encoder = models.ConvEncoder(stack=stack[:-1])
#     decoder = models.UpsampleDecoder(stack[:-1][::-1])
#     x, skips = encoder(x)

#     x = layers.ConvBlock(stack[-1])(x)

#     x = decoder(x, skips[::-1])
        
#     outputs = tf.keras.layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(x)

#     model = tf.keras.Model(inputs, outputs)
    
#     return model