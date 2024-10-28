import os
import json

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow import keras

from utils import optimizer_factory
from utils import callbacks as custom_callbacks

from dataloader.preprocessing import preprocess_for_train, preprocess_for_eval
from dataloader.dataset_factory import *
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, StopGradientLayer, has_intersection

def base_parse_record(record, record_prep_func=None):
    """Parse an ImageNet record from a serialized string Tensor."""

    parsed = record

    label = tf.reshape(parsed['label'], shape=[1])
    label = tf.cast(label, dtype=tf.int32)

    if record_prep_func is not None:
        image, label = record_prep_func(parsed['image'], label)

    # populate features and labels dict
    features = dict()
    labels = dict()
    features['image'] = image
    labels['label'] = label  

    features['is_tr_split'] = False
    features['cutmix_mask'] = tf.zeros((image.shape[0], image.shape[1], 1))

    return features, labels


def preprocess(
    image,
    label,
    is_training,
    image_size,
    mean_subtract=False,
    standardize=False,
    dtype="float32",
    one_hot=True,
    augmenter=None,
    num_classes=100,
    num_channels=3,
    prep_func_before_norm=None,
    prep_func_after_norm=None
    
    ):
    """Apply image preprocessing and augmentation to the image and label."""
    if is_training:
        image = preprocess_for_train(
          image,
          image_size=image_size,
          mean_subtract=mean_subtract,
          standardize=standardize,
          dtype=dtype,
          augmenter=augmenter,
          prep_func=prep_func_before_norm)
    else:
        image = preprocess_for_eval(
          image,
          image_size=image_size,
          num_channels=num_channels,
          mean_subtract=mean_subtract,
          standardize=standardize,
          dtype=dtype,
          prep_func=prep_func_before_norm)

    image = prep_func_after_norm(image)

    label = tf.cast(label, tf.int32)
    if one_hot:
        label = tf.one_hot(label, num_classes)
        label = tf.reshape(label, [num_classes])

    return image, label

def add_augmentation(model, image_size, train_batch_size=32, do_mixup=False, do_cutmix=False, custom_objects=None, update_batch_size=False):

    found = False
    for l in model.layers:
        if l.name == "mixup_weight":
            found = True
            break
    if found and not update_batch_size:
        return model

    if type(model) == keras.Sequential:
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape, name="seq_input")
        prev_layer = input_layer
        for layer in model.layers:
            layer._inbound_nodes = []
            prev_layer = layer(prev_layer)
        model = keras.models.Model([input_layer], [prev_layer])

    def cond_mixing(args):
      from dataloader.dataset_factory import mixing_lite
      images,mixup_weights,cutmix_masks,is_tr_split = args
      return tf.cond(tf.keras.backend.equal(is_tr_split[0],0), 
                     lambda: images, # eval phase
                     lambda: mixing_lite(images,mixup_weights,cutmix_masks, train_batch_size, do_mixup, do_cutmix)) # tr phase

    input_shape = (image_size, image_size, 3)  # Should handle any image size
    image_input = tf.keras.layers.Input(shape=input_shape, name="image")
    mixup_input = tf.keras.layers.Input(shape=(1, 1, 1), name="mixup_weight")
    cutmix_input = tf.keras.layers.Input(shape=(None, None, 1), name="cutmix_mask")
    is_tr_split = tf.keras.layers.Input(shape=(1), name="is_tr_split") # indicates whether we use tr or eval data loader
    inputs = [image_input,mixup_input,cutmix_input,is_tr_split]

    mixup_weights = inputs[1]
    cutmix_masks = inputs[2]
    is_tr_split = inputs[3]
    x = tf.keras.layers.Lambda(cond_mixing, name="input_lambda")([image_input,mixup_weights,cutmix_masks,is_tr_split])
          
    temp_model = tf.keras.Model(inputs=inputs, outputs=x)
    temp_model_dict = json.loads(temp_model.to_json())
    model_dict = json.loads(model.to_json())

    to_removed_names = []
    to_removed = []
    if found and update_batch_size:

        new_layer = None
        for layer in temp_model_dict["config"]["layers"]:
            if layer["name"] == "input_lambda":
                new_layer = layer
                break

        for layer in model_dict["config"]["layers"]:
            if layer["name"] == "input_lambda":
                layer["config"] = new_layer["config"] # replace
    else:

        for layer in model_dict["config"]["layers"]:
            if layer["class_name"] == "InputLayer":
                to_removed_names.append(layer["config"]["name"])
                to_removed.append(layer)

        for layer in model_dict["config"]["layers"]:
            for inbound in layer["inbound_nodes"]:
                if type(inbound[0]) == str:
                    if inbound[0] in to_removed_names:
                        inbound[0] = "input_lambda"
                else:
                    for ib in inbound:
                        if ib[0] in to_removed_names: # input
                            ib[0] = "input_lambda"

        for r in to_removed:
            model_dict["config"]["layers"].remove(r)

        model_dict["config"]["layers"] +=  temp_model_dict["config"]["layers"]
        model_dict["config"]["input_layers"] = [[layer.name, 0, 0] for layer in inputs]

    model_json = json.dumps(model_dict)
    if custom_objects is None:
        custom_objects = {}
    model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    for layer in model.layers:
        if layer.name in to_removed_names:
            continue
        model_.get_layer(layer.name).set_weights(layer.get_weights())

    return model_


def modify_model_gen(config, is_training=False):
    """Modify a model if it needs. (Input: Model)

    You can change your model upon your configuration. (ex. fp16)

    """
    image_size = config["image_size"]
    func = lambda model: add_augmentation(model, image_size)
    return func


def parse_record_gen(config, is_training=False, prep_func_before_norm=None, prep_func_after_norm=None):
    """ Parse a record from your tfrecord dataset.
    It includes a preprocessing function to process a raw image.

    (Input: tf.Record)

    Returns.
    
        a lambda function to parse a tf.Record

    """

    image_size = config["image_size"]

    prep = lambda image, label: preprocess(
        image,
        label,
        is_training=is_training,
        image_size=image_size,
        num_classes=config["num_classes"],
        prep_func_before_norm=prep_func_before_norm,
        prep_func_after_norm=prep_func_after_norm)
    parse_record = lambda record: base_parse_record(record, record_prep_func=prep)

    return parse_record


def preprocess_func_gen(config, is_training=False):
    """ Preprocess function executed in tf.data.Dataset right before calling prefetching.
    Data regularization methods such as `cutmix` and `mixup` may be called here.

    (Input: Data)

    Args.

        config: dict, a configuration dictionary

    Returns.

        a lambda function to handle a datum retrieved by your tf.dataset instance.

    """
    return None
