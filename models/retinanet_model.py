import tensorflow as tf
import layers

def _encode(x):
    x = layers.ConvBlock(num_filters=32)(x)
    return

def _decode():
    return

def _create_backbone():
    encoder = _encode()
    decoder = _decode()

    return decoder

def RetinaNet():
    _create_backbone()