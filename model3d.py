import tensorflow as tf
from keras import layers
from keras.models import Model

from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2

BIN, OVERLAP = 6, 0.1

# -------------------------
# ------- 3D MODEL --------
# -------------------------
def l2_normalize(x):
    # Compute the second norm for each (sin, cos) pare and normalize the values.
    # So (sin, cos) will be normalized into ( sin/sqrt(sin^2+cos^2) , cos/sqrt(sin^2+cos^2)).
    # Thus if the network gives (a,b), we are always sure that a^2 + b^2 = 1 and we can use arctan with no worries.
    return tf.nn.l2_normalize(x, axis=2)

# Construct the network
def build_model(input_shape=(224, 224, 3), weights=None, freeze=False, feature_extractor='vgg16'):
    if feature_extractor == 'mobilenetv2':
        feature_extractor_model = MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif feature_extractor == 'vgg16':
        feature_extractor_model = VGG16(include_top=False, weights=weights, input_shape=input_shape)
    else:
        print(
            "Requested a non-existing feature extractor model. Either choose from mobilenetv2 and vgg16 or add your own to the code")
        exit(-1)

    if freeze:
        for layer in feature_extractor_model.layers:
            layer.trainable = False

    x = layers.Flatten()(feature_extractor_model.output)

    dimension = layers.Dense(512)(x)
    dimension = layers.LeakyReLU(alpha=0.1)(dimension)
    dimension = layers.Dropout(0.5)(dimension)
    dimension = layers.Dense(3)(dimension)
    dimension = layers.LeakyReLU(alpha=0.1, name='dimension')(dimension)

    orientation = layers.Dense(256)(x)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Dropout(0.5)(orientation)
    orientation = layers.Dense(BIN * 2)(orientation)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Reshape((BIN, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    confidence = layers.Dense(256)(x)
    confidence = layers.LeakyReLU(alpha=0.1)(confidence)
    confidence = layers.Dropout(0.5)(confidence)
    confidence = layers.Dense(BIN, activation='softmax', name='confidence')(confidence)

    model = Model(feature_extractor_model.input, outputs=[dimension, orientation, confidence])

    return model
