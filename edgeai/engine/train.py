import os

import horovod.tensorflow.keras as hvd
import tensorflow as tf

from dataset_loader import load_tfdataset, TFHVDDatasetLoader
import callbacks as custom_callbacks

def get_datasets(config, parse_record_gen, batch_preprocess_func_gen):
    datasets = []
    batch_size = config["batch_size"]
    dataset = config["dataset"]
    dataset_splits = config["splits"]
    for split in dataset_splits[dataset]:
        ds = load_tfdataset(dataset, split)
        parse_record = parse_record_gen(config, split=="train")
        if batch_preprocess_func_gen is not None:
            prep_func = batch_preprocess_func_gen(config, split=="train")
        else:
            prep_func = None
        loader = TFHVDDatasetLoader(
            parse_record,
            batch_size,
            prep_func=prep_func,
            training=split=="train")
        datasets.append(loader.build(ds))

    return datasets

"""
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
"""

def train(
    config,
    model,
    model_name,
    epochs,
    parse_record_gen,
    get_optimizer_gen,
    get_loss_gen,
    get_callbacks_gen,
    get_metric_gen,
    get_model_callbacks_gen,
    batch_preprocess_func_gen=None,
    save_dir=None):

    batch_size = config["batch_size"]
    train_data_generator, valid_data_generator = get_datasets(config, parse_record_gen, batch_preprocess_func_gen)
    num_train_examples = config["num_train_examples"]
    num_val_examples = config["num_val_examples"]

    iters = num_train_examples // (batch_size * hvd.size())
    iters_val = num_val_examples // (batch_size * hvd.size())
    test_data_generator = valid_data_generator

    optimizer, learning_rate = get_optimizer_gen(config)(iters, epochs)
    callbacks = get_callbacks_gen(config)(model, optimizer, learning_rate)
    callbacks += get_model_callbacks_gen(config)(model, optimizer, learning_rate)
    loss = get_loss_gen(config)(model)
    metrics, monitor = get_metric_gen(config)(model)

    # compile
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False, experimental_run_tf_function=False)

    if save_dir is not None and hvd.local_rank() == 0:
        model_name_ = '%s_model.best.h5' % (model_name+"_"+config["dataset"])
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name_)

        if config["moving_average_decay"] > 0:
            mchk = custom_callbacks.AverageModelCheckpoint(update_weights=False,
                                          filepath=filepath,
                                          monitor=monitor,
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode="auto",
                                          save_freq="epoch")
            func = model.save
            model.save = lambda filepath, overwrite=True, options=None:\
                func(filepath, overwrite=overwrite, include_optimizer=False, options=options)

        else:
            mchk = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor=monitor,
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
                options=None,
            )
        callbacks.append(mchk)

    model_history = model.fit(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=callbacks,
                                    verbose=1 if hvd.local_rank() == 0 else 0,
                                    epochs=epochs,
                                    steps_per_epoch=iters * config["grad_accum_steps"],
                                    validation_steps=iters_val)

    del train_data_generator, valid_data_generator

    return model

def evaluate(model, valid_data_generator):
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=False)
    print(model.evaluate(valid_data_generator, verbose=1)[1])
