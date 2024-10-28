# Jong-Ryul Lee

import os
import sys
import json
import shutil
import time
import contextlib

import yaml
import tensorflow as tf

def update_meta(path, updates):
    with open(path, "r") as f:
        meta = json.load(f)

    for key, value in updates.items():
        assert key in meta
        meta[key] = value

    with open(path, "w") as f:
        json.dump(meta, f)

def get_value_from_meta(path, key):
    with open(path, "r") as f:
        meta = json.load(f)
    return meta[key]

class JSONLogging(object):

    def __init__(self, working_path):
        self.working_path = working_path

        self.paths = {
            "log":os.path.join(self.working_path, "log.out"),
            "meta":os.path.join(self.working_path, "meta.json"),
            "progress":os.path.join(self.working_path, "progress.json"),
            "values":os.path.join(self.working_path, "values.json")
        }

    def update(self, target, updates):
       
        with open(self.paths[target], "r") as f:
            struct = json.load(f)

        for key, value in updates.items():
            struct[key] = value

        with open(self.paths[target], "w") as f:
            json.dump(struct, f)       

    def add(self, target, updates):
        
        with open(self.paths[target], "r") as f:
            struct = json.load(f)

        for key, value in updates.items():
            if key not in struct:
                struct[key] = []
            # length of struct[key] is i
            struct[key].append([len(struct[key]) + 1, value])

        with open(self.paths[target], "w") as f:
            json.dump(struct, f)       


class RunningContext(object):

    def __init__(self, root_path, user_path, project_name, mode, identifier):
        self.root_path = root_path
        self.user_path = user_path
        self.project_name = project_name
        # self.project_dir = os.path.join(root_path, "projects", project_name)
        self.project_dir = os.path.join(user_path, project_name)
        self.mode = mode
        self.identifier = identifier
        self.job_dir = os.path.join(self.project_dir, "jobs", self.identifier)
        self.meta = None
        self.config = None

        # meta.json
        with open(os.path.join(self.job_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self.meta = meta

        # config.yaml
        with open(os.path.join(self.job_dir, "config.yaml"), 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                print("yaml error")
                sys.exit(1)

    def _model_load(self):
        model = tf.keras.load_model(model_path)

    def get_model_path(self):
        default_path =  os.path.join(self.job_dir, "model.h5")
        if os.path.exists(default_path):
            model_path = default_path
        else:
            model_path = self.meta["model_path"]
        assert os.path.exist(model_path)
        return model_path

    def get_model_path_from_name(self, identifier):
        with open(os.path.join(self.project_dir, "models", identifier, "meta.json"), "r") as f:
            meta = json.load(f)
        return os.path.join(self.user_path, meta["model_path"])

    def model_schema_copy(self, old, new):
        with open(os.path.join(old, "meta.json"), "r") as f:
            meta = json.load(f)
        meta["accuracy"] = -1
        meta["status"] = "new"
        meta["model_path"] = os.path.join(new, "model.h5")

        with open(os.path.join(new, "meta.json"), "w") as f:
            json.dump(meta, f)

        # assert not os.path.exists(os.path.join(new, "src"))
        shutil.copytree(os.path.join(old, "src"), os.path.join(new, "src"),  dirs_exist_ok=True)
       
    def train(self):
        o_model_path = self.get_model_path_from_name(self.meta["model_name"])
        n_model_path = os.path.join(self.job_dir, "before.h5")

        # training
        logger = JSONLogging(self.job_dir)
        with open(logger.paths["log"], "w") as f:
            with contextlib.redirect_stdout(f):
                shutil.copy(o_model_path, n_model_path) # backup
                for i in range(10):
                    logger.update("progress", {"progress":i*10.0})
                    logger.add("values", {"acc":i*11.0})
                    print(str(i)+" ...")
                    time.sleep(1)

        # after training
        trained_model_path = os.path.join(self.job_dir, "trained.h5")
        shutil.copy(o_model_path, trained_model_path) # trained model in a job folder.

        # make a trained model
        new_model_name = self.meta["model_name"] + "_trained_" + self.meta["tag"]
        new_dir = os.path.join(self.project_dir, "models", new_model_name)
        new_model_path = os.path.join(new_dir, "model.h5")
       
        assert not os.path.exists(new_dir)
        os.mkdir(new_dir)

        # copy base code
        old_dir = os.path.join(self.project_dir, "models", self.meta["model_name"])
        self.model_schema_copy(old_dir, new_dir)
        shutil.copy(trained_model_path, new_model_path)

        # meta update
        update_meta(os.path.join(new_dir, "meta.json"), {"accuracy":55.0}) # dummy value

    def evaluate(self):

        # evaluate model
        # ...

        # meta update
        model_dir = os.path.join(self.project_dir, "models", self.meta["model_name"])
        update_meta(os.path.join(model_dir, "meta.json"), {"accuracy":18.0}) # dummy value

    def optimize(self):
        o_model_path = self.get_model_path_from_name(self.meta["model_name"])
        n_model_path = os.path.join(self.job_dir, "before.h5")

        # optimization
        # (dummy)
        shutil.copy(o_model_path, n_model_path) # backup
        logger = JSONLogging(self.job_dir)
        with open(logger.paths["log"], "w") as f:
            with contextlib.redirect_stdout(f):
                for i in range(10):
                    logger.update("progress", {"progress":i*11.0})
                    logger.add("values", {"acc":i*12.0})
                    print(str(i)+" ...")
                    time.sleep(1)

        # after optimization
        opt_model_path = os.path.join(self.job_dir, "optimized.h5")
        shutil.copy(o_model_path, opt_model_path) # trained model in a job folder.

        # make a trained model
        new_model_name = self.meta["model_name"] + "_" + self.meta["method"] + "_" + self.meta["tag"]
        new_dir = os.path.join(self.project_dir, "models", new_model_name)
        new_model_path = os.path.join(new_dir, "model.h5")
       
        assert not os.path.exists(new_dir)
        os.mkdir(new_dir)

        # copy base code
        old_dir = os.path.join(self.project_dir, "models", self.meta["model_name"])
        self.model_schema_copy(old_dir, new_dir)
        shutil.copy(opt_model_path, new_model_path)

        # meta update
        update_meta(os.path.join(new_dir, "meta.json"), {"accuracy":40.0, "inference_time":10.0}) # dummy value

    def profile(self):

        o_model_path = self.get_model_path_from_name(self.meta["model_name"])

        # profile
        # ...
        logger = JSONLogging(self.job_dir)
        with open(logger.paths["log"], "w") as f:
            with contextlib.redirect_stdout(f):
                for i in range(10):
                    logger.update("progress", {"progress":i*13.0})
                    logger.add("values", {"acc":i*14.0})
                    print(str(i)+" ...")
                    time.sleep(1)

        # meta update
        model_dir = os.path.join(self.project_dir, "models", self.meta["model_name"])
        update_meta(os.path.join(model_dir, "meta.json"), {"inference_time":10.0}) # dummy value

    def search(self):
        
        # trained_model_path
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        model = EfficientNetB0(weights='imagenet')
        logger = JSONLogging(self.job_dir)
        with open(logger.paths["log"], "w") as f:
            with contextlib.redirect_stdout(f):
                for i in range(10):
                    logger.update("progress", {"progress":i*14.0})
                    logger.add("values", {"acc":i*15.0})
                    print(str(i)+" ...")
                    time.sleep(1)

        # make a trained model
        new_model_name = self.meta["model_name"]
        new_dir = os.path.join(self.project_dir, "models", new_model_name)
        new_model_path = os.path.join(new_dir, "model.h5")
      
        # assert not os.path.exists(new_dir)
        # os.mkdir(new_dir)

        # copy base code
        old_dir = os.path.join(self.root_path, "template/project", "models", "model")
        self.model_schema_copy(old_dir, new_dir)
        tf.keras.models.save_model(model, new_model_path)

        # meta update
        update_meta(os.path.join(new_dir, "meta.json"), {"accuracy":11.0}) # dummy value

    def query(self):

        o_model_path = self.get_model_path_from_name(self.meta["model_name"])
        n_model_path = os.path.join(self.job_dir, "before.h5")

        # optimization
        # (dummy)
        shutil.copy(o_model_path, n_model_path) # backup
        logger = JSONLogging(self.job_dir)
        with open(logger.paths["log"], "w") as f:
            with contextlib.redirect_stdout(f):
                for i in range(10):
                    logger.update("progress", {"progress":i*15.0})
                    logger.add("values", {"acc":i*16.0})
                    print(str(i)+" ...")
                    time.sleep(1)

        # after optimization
        opt_model_path = os.path.join(self.job_dir, "optimized.h5")
        shutil.copy(o_model_path, opt_model_path) # trained model in a job folder.

        # make a trained model
        new_model_name = self.meta["model_name"] + "_" + self.meta["method"] + "_" + self.meta["tag"]
        new_dir = os.path.join(self.project_dir, "models", new_model_name)
        new_model_path = os.path.join(new_dir, "model.h5")
       
        assert not os.path.exists(new_dir)
        os.mkdir(new_dir)

        # copy base code
        old_dir = os.path.join(self.project_dir, "models", self.meta["model_name"])
        self.model_schema_copy(old_dir, new_dir)
        shutil.copy(opt_model_path, new_model_path)

        # meta update
        update_meta(os.path.join(new_dir, "meta.json"), {"accuracy":33.0, "inference_time":15.0}) # dummy value

    def execute(self):

        if self.mode == "train":
            self.train()

        elif self.mode == "evaluate":
            self.evaluate()

        elif self.mode == "optimize":
            self.optimize()

        elif self.mode == "profile":
            self.profile()

        elif self.mode == "search":
            self.search()
        
        elif self.mode == "query":
            self.query()
        else:
            raise NotImplementedError()

def view(model_path):
    """
    import importlib.util
    modelholder = os.path.join(root_path, "projects", project_name, "models", model_name)
    spec = importlib.util.spec_from_file_location("model_aux", os.path.join(modelholder, "src/model.py"))
    model_aux = importlib.util.module_from_spec(spec)
    sys.modules["model_aux"] = model_aux
    spec.loader.exec_module(model_aux)
    model_aux.prior()
    
    meta_path = os.path.join(modelholder, "meta.json")
    model_path = get_value_from_meta(meta_path, "model_path")
    #model = tf.keras.models.load_model(model_path)
    """

    from tensorflow.keras.applications.efficientnet import EfficientNetB0
    model = EfficientNetB0(weights='imagenet')
    model_json = model.to_json()
    model_dict = json.loads(model_json)
   
    view_dict = {"layers":[]}
    layers = model_dict["config"]["layers"]
    for idx, layer in enumerate(layers):
        shape = [
            int(s) if s is not None else "?" for s in model.get_layer(layer["name"]).output.shape
        ]
        layer_dict = {
            "name":layer["name"],
            "classname":layer["class_name"],
            "idx":idx,
            "output_shape":[shape],
            "inbound":[]
        }
        for inbound in layer["inbound_nodes"]:
            if type(inbound[0]) == str:
                layer_dict["inbound"].append([inbound[0], 0])
            else:
                for ib in inbound:
                    layer_dict["inbound"].append([ib[0], ib[1]])
        view_dict["layers"].append(layer_dict)

    output_layer = {
        "name":"OUTPUT",
        "classname":"OUTPUT",
        "idx":len(layers),
        "output_shape": None,
        "inbound":[[view_dict["layers"][-1]["name"], 0]]
    }
    view_dict["layers"].append(output_layer)

    # dir_ = os.path.dirname(model_path)
    # basename = os.path.basename(model_path)
    # view_file_path = os.path.join(dir_, basename+".json")
    view_file_path = os.path.join(model_path, "view.json")
    with open(view_file_path, "w") as f:
        json.dump(view_dict, f, indent=4)

def run():

    mode = sys.argv[1]
    # root_path = "/home/jongryul/work/package_20221121/2_web_gui/edgeai"
    root_path = os.getcwd()
    root_path = "/".join(root_path.split("/")[0:-1])

    if mode == "test":
        project_name = "project_dummy"
        test(root_path, project_name)
    elif mode == "cleanup":
        project_name = "project_dummy"
        clean_up(root_path, project_name)
    elif mode == "view":
        model_path = sys.argv[2]
        view(model_path)
    else:
        job_path = sys.argv[2]
        # f"{cwd}/edgeai/users/{session['user']}/{data['project_name']}/jobs/{ms}"

        project_name = job_path.split("/")[-3]
        user_path = "/".join(job_path.split("/")[0:-3])
        root_path = "/".join(job_path.split("/")[0:-5])
        identifier = os.path.basename(job_path)
      
        cxt = RunningContext(root_path, user_path, project_name, mode, identifier)
        cxt.execute()

def test(root_path, project_name):

    if os.path.exists(root_path+"/projects/project_dummy"):
        shutil.rmtree(root_path+"/projects/project_dummy")
    shutil.copytree(root_path+"/template/project_dummy", root_path+"/projects/project_dummy")

    # train
    mode = "train"
    identifier = "job1"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_train", job_dir)
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model1"})

    cmd = "python " + root_path + "/cli/cli_dummy.py train " + job_dir
    os.system(cmd)
    print(cmd)

    # evaluate
    mode = "evaluate"
    identifier = "job2"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_eval", job_dir)
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model1_trained_tag"})

    cmd = "python " + root_path + "/cli/cli_dummy.py evaluate " + job_dir
    os.system(cmd)
    print(cmd)

    # optimize
    mode = "optimize"
    identifier = "job3"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_optimize", job_dir)
    shutil.copy(root_path+"/template/methods/opt/bespoke/config.yaml", os.path.join(job_dir, "config.yaml"))
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model1_trained_tag"})

    cmd = "python " + root_path + "/cli/cli_dummy.py optimize " + job_dir
    os.system(cmd)
    print(cmd)

    # profile
    mode = "profile"
    identifier = "job4"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_profile", job_dir)
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model1_trained_tag_bespoke_tag"})

    cmd = "python " + root_path + "/cli/cli_dummy.py profile " + job_dir
    os.system(cmd)
    print(cmd)

    # search
    mode = "search"
    identifier = "job5"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_search", job_dir)
    shutil.copy(root_path+"/template/methods/search/test/config.yaml", os.path.join(job_dir, "config.yaml"))
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model_searched"})

    cmd = "python " + root_path + "/cli/cli_dummy.py search " + job_dir
    os.system(cmd)
    print(cmd)

    # query
    mode = "query"
    identifier = "job6"
    job_dir = root_path+"/projects/"+project_name+"/jobs/"+identifier
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    shutil.copytree(root_path+"/template/project/jobs/job_query", job_dir)
    shutil.copy(root_path+"/template/methods/opt/bespoke/config.yaml", os.path.join(job_dir, "config.yaml"))
    update_meta(os.path.join(job_dir, "meta.json"), {"model_name":"model1_trained_tag_bespoke_tag"})

    cmd = "python " + root_path + "/cli/cli_dummy.py query " + job_dir
    os.system(cmd)
    print(cmd)


def clean_up(root_path, project_name):

    if os.path.exists(root_path+"/projects/project_dummy"):
        shutil.rmtree(root_path+"/projects/project_dummy")

if __name__ == "__main__":
    run()
