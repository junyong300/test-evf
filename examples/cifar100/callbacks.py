import horovod.tensorflow.keras as hvd
import tensorflow as tf

from utils import callbacks as custom_callbacks

def get_callbacks_gen(config):
    return lambda model, optimizer, learning_rate: get_callbacks(config, model, optimizer, learning_rate)

def get_callbacks(config, model, optimizer, learning_rate):

    callbacks = []

    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())

    class StepCounter(tf.keras.callbacks.Callback):

        def __init__(self, scheduler):
            super(StepCounter, self).__init__()
            self.scheduler = scheduler
            self._counter = 1

        def on_train_batch_begin(self, batch, logs=None):
            self._counter += 1

        def on_epoch_begin(self, epoch, logs=None):
            print(self.scheduler(self._counter))

    if hvd.rank() == 0:
        counter_ = StepCounter(learning_rate)
        callbacks.append(counter_)

    if config["moving_average_decay"] > 0:
        callbacks.append(
            custom_callbacks.MovingAverageCallback(intratrain_eval_using_ema=config["intratrain_eval_using_ema"]))

    return callbacks
