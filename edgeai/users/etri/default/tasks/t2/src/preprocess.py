"""Template for prep.

"""

def modify_model_if_needed(config):
    """Modify a model if it needs. (Input: Model)

    You can change your model upon your configuration. (ex. fp16)

    """
    return None


def parse_record_gen(config):
    """ Parse a record from your tfrecord dataset.
    It includes a preprocessing function to process a raw image.

    (Input: tf.Record)

    Returns.
    
        a lambda function to parse a tf.Record

    """
    return None

def preprocess_func_gen(config):
    """ Preprocess function executed in tf.data.Dataset right before calling prefetching.
    Data regularization methods such as `cutmix` and `mixup` may be called here.

    (Input: Data)

    Args.

        config: dict, a configuration dictionary

    Returns.

        a lambda function to handle a datum retrieved by your tf.dataset instance.

    """
    return None
