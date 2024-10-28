import functools

from dataloader.dataset_factory import mixing

def model_preprocess_func_gen(config, is_training=False):
    """
        Return a lambda function which is the final preprocessing function for the model.
        The input of this function is supposed to be the same as the data of the provided dataset.

    """

    return functools.partial(mixing, 32, 0.0, 0.0, True)
