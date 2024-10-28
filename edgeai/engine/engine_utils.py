
import inspect
import os
import shutil

RESERVED = ["config", "is_training"]

def fill_func(func, model_module):
    args = inspect.getfullargspec(func)[0]
    add_args = []
    for arg in args:
        if arg in RESERVED:
            continue

        func_arg = getattr(model_module, arg)
        add_args.append(func_arg)

    new_func = lambda config, is_training=False: func(config, is_training, *add_args)
    return new_func

task_mapper = {
    "train": "trained",
    "optimize": "optimized",
    "search": "method",
    "query": "method"
}
def check_dir(user_path, project_name, model_name, mode, tag=None, method=None, overwrite=False):

    if mode not in task_mapper:
        return None

    project_dir = os.path.join(user_path, project_name)
    if tag is None: # search
        new_model_name = model_name
    else:
        if method is not None:
            new_model_name = model_name + "_" + method + "_" + tag
        else:
            new_model_name = model_name + "_" + task_mapper[mode] + "_" + tag
    new_dir = os.path.join(project_dir, "models", new_model_name)
    
    if os.path.exists(new_dir):
        if not overwrite:
            return None
        else:
            shutil.rmtree(new_dir)
    return new_dir
