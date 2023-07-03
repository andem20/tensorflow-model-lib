import tensorflow as tf
import models

class RetinaNet(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model, decoder: tf.keras.Model, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.fpn = FPN(backbone, decoder)
        self.num_anchor_boxes = 9
        self.box_heads = []
        self.class_heads = []
        self.num_classes = num_classes

        for output_shape in self.fpn.output_shape:
            self.box_heads.append(MLP(input_shape=output_shape[1:], num_outputs=self.num_anchor_boxes * 4, name_prefix="box"))
            self.class_heads.append(MLP(input_shape=output_shape[1:], num_outputs=self.num_anchor_boxes * num_classes, name_prefix="class"))
            
        self.out = self.call(backbone.inputs)
        super().__init__(inputs=backbone.inputs, outputs=self.out)

    def call(self, inputs, training=None, mask=None):
        class_outputs = []
        box_outputs = []
        features = self.fpn(inputs)
        N = tf.shape(inputs)[0]
        for feature, box_head, class_head in zip(features, self.box_heads, self.class_heads):
            box_predictions = box_head(feature)
            box_outputs.append(tf.reshape(box_predictions, shape=(N, -1, 4)))
            class_predictions = class_head(feature)
            class_outputs.append(tf.reshape(class_predictions, shape=(N, -1, self.num_classes)))

        box_outputs = tf.concat(box_outputs, axis=0)
        class_outputs = tf.concat(class_outputs, axis=0)

        return tf.concat([box_outputs, class_outputs], axis=-1)
    
class FPN(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model, decoder: tf.keras.Model):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.out = self.call(backbone.inputs)
        super().__init__(inputs=backbone.inputs, outputs=self.out)

    def call(self, inputs, training=None, mask=None):
        inputs, skips = self.backbone(inputs)
        return self.decoder(inputs)
    
class MLP(tf.keras.Model):
    def __init__(self, input_shape: tuple[int, int, int], num_outputs: int, num_conv_layers: int = 4, num_conv_filters: int = 256, name_prefix: str = ""):
        super().__init__()
        self.num_outputs = num_outputs
        self.input_layer = tf.keras.Input(shape=input_shape)
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(num_conv_filters, 3, activation="relu", padding="same") for _ in range(num_conv_layers)
        ])
        self.output_layer = tf.keras.layers.Conv2D(num_outputs, 1, padding="same", activation="relu")
        self.out = self.call(self.input_layer)
        self._name = f"{name_prefix}_{self._name}"

    def call(self, inputs):
        inputs = self.conv_layers(inputs)
        return self.output_layer(inputs)