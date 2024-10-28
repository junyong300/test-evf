def get_metric(model, config):
    """Get loss upon your configuration

        Args.

            model: model
            config: dict

        Returns

            metrics, monitor

    """

    return ["accuracy"], "val_accuracy"

def get_metric_gen(config):
    return lambda model: get_metric(model, config)
