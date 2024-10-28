from __future__ import print_function

import hashlib
import time
import os
import traceback
import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml
import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization

import model_handler
from edgeai.engine.utils import fill_func
from edgeai.engine.train import train, get_datasets, evaluate

from loss import get_loss_gen
from callbacks import get_callbacks_gen
from metric import get_metric_gen
from optimizer import get_optimizer_gen
from model import get_model_instance
from preprocess import parse_record_gen, modify_model_gen
from model_prep import model_preprocess_func_gen
from model_callbacks import get_model_callbacks_gen

def run():

    hvd.init()

    config_ = "config.yaml"

    # load config
    with open(config_, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for key in config:
        if config[key] in ["true", "false"]: # boolean handling.
            config[key] = config[key] == "true"

    model = get_model_instance(config, None)
    model = modify_model_gen(config, is_training=True)(model)

    filled_parse_record_gen = fill_func(parse_record_gen, model_handler)

    datasets = get_datasets(config, parse_record_gen=filled_parse_record_gen, batch_preprocess_func_gen=model_preprocess_func_gen)

    train(
        config,
        model,
        "efnet",
        1,
        parse_record_gen=filled_parse_record_gen,
        get_optimizer_gen=get_optimizer_gen,
        get_loss_gen=get_loss_gen,
        get_callbacks_gen=get_callbacks_gen,
        get_metric_gen=get_metric_gen,
        get_model_callbacks_gen=get_model_callbacks_gen,
        batch_preprocess_func_gen=model_preprocess_func_gen,
        save_dir=None)

    evaluate(model, datasets[1])


if __name__ == "__main__":

    run()
