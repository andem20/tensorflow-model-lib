import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters: int, kernel_size = 3):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.layers = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs):
        return self.layers(inputs)