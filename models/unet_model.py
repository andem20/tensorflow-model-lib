import tensorflow as tf
import layers

def _encode(x, filters):
    x = layers.ConvBlock(filters)(x)
    max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    return x, max_pooling

def _decode(x, skip, filters):
    x = layers.UpsampleBlock(filters)(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = layers.ConvBlock(filters)(x)
    
    return x

def UNet(num_classes, input_shape, stack=[32, 64, 128, 256, 512]):
    skips = {}
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    for i, filters in enumerate(stack[:-1]):
        skips[i], x = _encode(x, filters)

    x = layers.ConvBlock(stack[-1])(x)
        
    for i, filters in enumerate(reversed(stack[:-1])):
        x = _decode(x, skips[(len(stack) - 2) - i], filters)
        
    outputs = tf.keras.layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model