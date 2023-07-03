import tensorflow as tf
import layers

class ConvEncoder(tf.keras.Model):
    """
    CNN encoder taking a stack of conv layers as a list of the filter size.
    >>> ConvEncoder(stack=[32, 64, 128])
    """
    def __init__(self, stack: list[int], input_shape=(256, 256, 3)):
        super(ConvEncoder).__init__()
        self.input_layer = tf.keras.Input(shape=input_shape)
        self.layer_stack: list[tf.keras.layers.Layer] = [None] * len(stack)
        self.max_pooling = [None] * len(stack)
        self.stack = stack
    
        for i, num_filters in enumerate(stack):
            self.layer_stack[i] = layers.ConvBlock(num_filters=num_filters)
            self.max_pooling[i] = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.out = self.call(self.input_layer)
        super().__init__(inputs=self.input_layer, outputs=self.out)

    def call(self, inputs, training=None):
        for i, layer in enumerate(self.layer_stack):
            inputs = layer(inputs)
            inputs = self.max_pooling[i](inputs)

        return inputs