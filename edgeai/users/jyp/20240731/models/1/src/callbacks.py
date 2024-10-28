
def get_model_callbacks_gen(config):
    return lambda model, optimizer, learning_rate: get_callbacks(config, model, optimizer, learning_rate)

def get_callbacks(config, model, optimizer, learning_rate):
    return []
