import tensorflow as tf
import layers

class ConvEncoder(tf.keras.Model):
    def __init__(self, stack: list[int]):
        super().__init__()
        self.layer_stack: list[tf.keras.layers.Layer] = [None] * len(stack)
        self.max_pooling = [None] * len(stack)
        self.stack = stack
        # self.skips = tf.TensorArray(dtype=tf.float32, size=len(stack))
        
        for i, num_filters in enumerate(stack):
            self.layer_stack[i] = layers.ConvBlock(num_filters=num_filters)
            self.max_pooling[i] = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        skips = [None] * len(self.stack)
        for i, layer in enumerate(self.layer_stack):
            skips[i] = layer(inputs)
            inputs = self.max_pooling[i](skips[i])

        return inputs, skips