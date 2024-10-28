def get_loss(config, model):
    """Get loss upon your configuration

        Args.

            config: dict

        Returns

            A loss

    """

    return "categorical_crossentropy"

def get_loss_gen(config):
    return lambda model:get_loss(config, model)
