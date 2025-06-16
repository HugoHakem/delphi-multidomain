import os
import mlflow
import yaml
import json
import time
from glob import glob
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # primero seteás el tracking URI
client = MlflowClient()  # después creás el cliente

MLRUNS_DIR = "./mlruns"

def migrate_run(run_dir, experiment_id_map):
    with open(os.path.join(run_dir, "meta.yaml")) as f:
        meta = yaml.safe_load(f)
    old_exp_id = meta["experiment_id"]
    run_name = meta["run_name"]
    start_time = meta.get("start_time", int(time.time() * 1000))
    end_time = meta.get("end_time", start_time + 1000)

    exp_name = f"experiment_{old_exp_id}"

    print("\n"*10)
    print(f"{client.search_experiments()=}")
    print(f"{experiment_id_map=}")
    if exp_name not in experiment_id_map:
        existing = client.get_experiment_by_name(exp_name)
        if existing is not None:
            new_exp_id = existing.experiment_id
        else:
            new_exp_id = mlflow.create_experiment(exp_name)
        experiment_id_map[exp_name] = new_exp_id
    else:
        new_exp_id = experiment_id_map[exp_name]


    run = client.create_run(
        experiment_id=new_exp_id,
        start_time=start_time,
        tags={"mlflow.runName": run_name}
    )
    run_id = run.info.run_id

    from mlflow.entities import Metric, Param, RunTag
    
    # Parámetros
    params = []
    param_dir = os.path.join(run_dir, "params")
    if os.path.exists(param_dir):
        for f in os.listdir(param_dir):
            with open(os.path.join(param_dir, f)) as pf:
                params.append(Param(key=f, value=pf.read()))
    
    # Métricas
    metrics = []
    metric_dir = os.path.join(run_dir, "metrics")
    if os.path.exists(metric_dir):
        for f in os.listdir(metric_dir):
            with open(os.path.join(metric_dir, f)) as mf:
                for line in mf:
                    step, ts, value = line.strip().split()
                    metrics.append(Metric(key=f, value=float(value), timestamp=int(float(ts)), step=int(step)))
    
    # Tags
    tags = []
    tag_dir = os.path.join(run_dir, "tags")
    if os.path.exists(tag_dir):
        for tag_file in os.listdir(tag_dir):
            with open(os.path.join(tag_dir, tag_file)) as tf:
                tags.append(RunTag(key=tag_file, value=tf.read()))
    
    # Logueo en batch
    client.log_batch(run_id, metrics=metrics, params=params, tags=tags)

    client.set_terminated(run_id, status="FINISHED", end_time=end_time)

def migrate_all():
    experiment_id_map = {}
    for exp_dir in glob(os.path.join(MLRUNS_DIR, "[0-9]*")):
        for run_dir in glob(os.path.join(exp_dir, "*")):
            if os.path.isdir(run_dir):
                print(f"Migrating {exp_dir} / {run_dir}")
                migrate_run(run_dir, experiment_id_map)

if __name__ == "__main__":
    migrate_all()

