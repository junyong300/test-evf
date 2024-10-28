import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization


""" Model-related functions.

"""

def get_custom_objects():
    pass

def get_model_instance(config, model_path=None):
    """This function returns a model instance upon the input config.

    Args.
        
        config: dict, the dictionary defined by `config.yaml`.

    Returns.

        A model instance.

    """

    efnb0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, input_tensor=None, weights=None,
        input_shape=(224, 224, 3), pooling=None, classes=100,
        classifier_activation='softmax'
    )
    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    return model
