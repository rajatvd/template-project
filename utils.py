from sacred.observers import FileStorageObserver
import incense
import json


def get_run_dir(ex):
    """Helper for sacred experiment logging."""
    for obs in ex.observers:
        if type(obs) == FileStorageObserver:
            return obs.dir

    return '.'


def get_incense_loader(authfile):
    with open(authfile, "r") as f:
        js = json.loads(f.read())
    db_name = js["db_name"]
    ck = js['client_kwargs']
    mongo_uri = f"mongodb://{ck['username']}:{ck['password']}@{ck['host']}:{ck['port']}/{db_name}\?authSource=admin"
    # print(mongo_uri)
    return incense.ExperimentLoader(mongo_uri=mongo_uri, db_name=db_name)
