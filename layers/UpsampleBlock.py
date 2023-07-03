import tensorflow as tf

class UpsampleBlock(tf.keras.layers.Layer):
    """
    Upsample block creating two Conv2dTranspose layers.
    """
    def __init__(self, num_filters: int, kernel_size = 3):
        super(UpsampleBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.layers = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, activation="relu", padding="same"),
            tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, activation="relu", padding="same", strides=2)
        ])

    def call(self, inputs):  
        return self.layers(inputs)