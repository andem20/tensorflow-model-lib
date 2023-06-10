import tensorflow as tf
import layers

class UpsampleDecoder(tf.keras.Model):
    def __init__(self, stack: list[int]):
        super(UpsampleDecoder, self).__init__()
        self.stack = stack
        self.up_layers_stack: list[tf.keras.layers.Layer] = []
        self.conv_layers_stack: list[tf.keras.layers.Layer] = []

        for num_filters in stack:
            self.up_layers_stack.append(layers.UpsampleBlock(num_filters=num_filters))
            self.conv_layers_stack.append(layers.ConvBlock(num_filters))

    def call(self, inputs, skips):
        for i, layer in enumerate(self.up_layers_stack):
            inputs = layer(inputs)
            inputs = tf.keras.layers.Concatenate()([inputs, skips[i]])
            inputs = self.conv_layers_stack[i](inputs)
            
        return inputs