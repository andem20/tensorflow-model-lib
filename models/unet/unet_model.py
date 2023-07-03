import tensorflow as tf
import layers
import models

def UNet(encoder: tf.keras.Model, skips: list[tf.keras.layers.Layer], num_classes: int):
    """
    UNet taking an encoder and creates a decoder from specified skip connections.
    >>> encoder = models.ConvEncoder(stack=[32, 64, 128, 256], input_shape=(64, 64, 3)))
    >>> skips = encoder.layer_stack
    >>> models.UNet(encoder, skips=encoder.layer_stack, num_classes=3)
    """
    x = encoder.output

    for i, skip in enumerate(skips[::-1]):
        skip_out = skip.output
        num_filters = skip.get_output_shape_at(0)[3]
        x = layers.UpsampleBlock(num_filters=num_filters)(x)
        x = tf.keras.layers.Concatenate()([x, skip_out])

    outputs = tf.keras.layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=outputs)