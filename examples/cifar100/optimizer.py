import horovod.tensorflow.keras as hvd
import tensorflow as tf

from utils import optimizer_factory

def get_optimizer_gen(config):
    return lambda iters, epochs: get_optimizer(config, iters, epochs)

def get_optimizer(config, iters, epochs):
    """Return your optimizer upon your configuration

    Args.

        config: dict

    Returns

        An optimizer instance

    """

    lr_params = {
        'name':config["lr_name"],
        'initial_lr': config["initial_lr"],
        'decay_epochs': config["decay_epochs"],
        'decay_rate': config["decay_rate"],
        'warmup_epochs': config["warmup_epochs"],
        'examples_per_epoch': None,
        'boundaries': None,
        'multipliers': None,
        'scale_by_batch_size': 0.0,
        'staircase': True,
        't_mul': config["t_mul"],
        'm_mul': config["m_mul"],
        'alpha': config['alpha']
    }

    batch_size = config["batch_size"]

    num_train_examples = config["num_train_examples"]
    learning_rate = optimizer_factory.build_learning_rate(
        params=lr_params,
        batch_size=batch_size * hvd.size() * config["grad_accum_steps"], # updates are iteration based not batch-index based
        train_steps=iters,
        max_epochs=epochs)

    opt_params = {
        'name': config["opt_name"],
        'decay': config["decay"],
        'epsilon': config["epsilon"],
        'momentum': config["momentum"],
        'lookahead': config["lookahead"],
        'moving_average_decay': config["moving_average_decay"],
        'nesterov': config["nesterov"],
        'beta_1': config["beta_1"],
        'beta_2': config["beta_2"],

    }

    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=config["opt_name"],
        base_learning_rate=learning_rate,
        params=opt_params
    )

    if config["use_amp"] and config["grad_accum_steps"] > 1:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    elif config["grad_accum_steps"] == 1:
        optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16 if config["hvd_fp16_compression"] else hvd.Compression.none)

    return optimizer, learning_rate
