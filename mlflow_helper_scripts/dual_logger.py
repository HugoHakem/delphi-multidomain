# NOTE: DualMLflowLogger updated with efficient log_batch implementation

import uuid
import os
import time
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from mlflow.entities import Metric


class DualMLflowLogger:
    def __init__(self, local_uri, remote_uri, experiment_name, artifact_size_limit_mb=10):
        self.local_client = MlflowClient(tracking_uri=local_uri)
        self.remote_client = MlflowClient(tracking_uri=remote_uri)
        self.artifact_size_limit_bytes = artifact_size_limit_mb * 1024 * 1024

        self.local_exp_id = self._get_or_create_experiment(self.local_client, experiment_name)
        self.local_run = self.local_client.create_run(self.local_exp_id)
        self.local_run_id = self.local_run.info.run_id

        self.shared_run_id = str(uuid.uuid4())
        if "dagshub.com" in remote_uri:
            dagshub.init(repo_name=experiment_name, repo_owner="rbonazzola", mlflow=True)
            mlflow.start_run()
            self.remote_run_id = mlflow.active_run().info.run_id
        else:
            self.remote_exp_id = self._get_or_create_experiment(self.remote_client, experiment_name, allow_create=False)
            self.remote_run = self.remote_client.create_run(self.remote_exp_id)
            self.remote_run_id = self.remote_run.info.run_id

        self.set_tag("shared_run_id", self.shared_run_id)
        self.set_tag("run_name", f"Run {self.shared_run_id}")

    def _get_or_create_experiment(self, client, name, allow_create=True):
        try:
            exp = client.get_experiment_by_name(name)
            return exp.experiment_id if exp else (
                client.create_experiment(name) if allow_create else "0"
            )
        except Exception as e:
            print(f"[DualLogger] ⚠️ Fallback to default experiment due to error: {e}")
            return "0"

    def log_metric(self, key, value, step=None):
        self.local_client.log_metric(self.local_run_id, key, value, step=step)
        mlflow.log_metric(key, value, step=step)

    def log_param(self, key, value):
        self.local_client.log_param(self.local_run_id, key, value)
        mlflow.log_param(key, value)

    def log_artifact(self, local_path, artifact_path=None):
        self.local_client.log_artifact(self.local_run_id, local_path, artifact_path)

        file_size = os.path.getsize(local_path)
        if file_size <= self.artifact_size_limit_bytes:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"[DualLogger] ⏭️ Skipped remote logging: {local_path} ({file_size/1024**2:.2f} MB)")

    def set_tag(self, key, value):
        self.local_client.set_tag(self.local_run_id, key, value)
        mlflow.set_tag(key, value)

    def log_batch(self, metrics: dict[str, list[tuple[int, float]]]):
        timestamp = int(time.time() * 1000)
        # Prepare local Metric objects
        metric_objs = []
        for key, values in metrics.items():
            for step, value in values:
                metric_objs.append(Metric(key=key, value=value, step=step, timestamp=timestamp))
        # Log in batch to local backend
        self.local_client.log_batch(run_id=self.local_run_id, metrics=metric_objs)
        # Log in batch to remote backend (e.g., DagsHub)
        self.remote_client.log_batch(run_id=self.remote_run_id, metrics=metric_objs)

    def get_run_id(self):
        return self.local_run_id

    def get_shared_run_id(self):
        return self.shared_run_id





# import uuid
# import os
# import time
# import mlflow
# from mlflow.tracking import MlflowClient
# import dagshub
# from mlflow.entities import Metric
# 
# 
# class DualMLflowLogger:
#     def __init__(self, local_uri, remote_uri, experiment_name, artifact_size_limit_mb=10):
#         # Set up MLflow clients for local and remote backends
#         self.local_client = MlflowClient(tracking_uri=local_uri)
#         self.remote_client = MlflowClient(tracking_uri=remote_uri)
#         self.artifact_size_limit_bytes = artifact_size_limit_mb * 1024 * 1024
# 
#         # Get or create experiments in both backends
#         self.local_exp_id = self._get_or_create_experiment(self.local_client, experiment_name)
#         self.remote_exp_id = self._get_or_create_experiment(self.remote_client, experiment_name, allow_create=False)
#       
#         self.local_run = self.local_client.create_run(self.local_exp_id)
#         self.local_run_id = self.local_run.info.run_id
#       
#         # self.remote_client.create_run(self.remote_exp_id)
#         # self.remote_run_id = self.remote_run.info.run_id
# 
#         # Create a shared run_id and initialize runs in both backends
#         self.shared_run_id = str(uuid.uuid4())
# 
#         if "dagshub.com" in remote_uri:
#             dagshub.init(repo_name=experiment_name, repo_owner="rbonazzola", mlflow=True)
#             mlflow.start_run()
#             self.remote_run_id = mlflow.active_run().info.run_id
#         else:
#             self.remote_exp_id = self._get_or_create_experiment(self.remote_client, experiment_name, allow_create=False)
#             self.remote_run = self.remote_client.create_run(self.remote_exp_id)
#             self.remote_run_id = self.remote_run.info.run_id
# 
#         self.set_tag("shared_run_id", self.shared_run_id)
#         # self.remote_client.set_tag(self.remote_run_id, "shared_run_id", shared_id)
# 
# 
#     def _get_or_create_experiment(self, client, name, allow_create=True):
#         try:
#             exp = client.get_experiment_by_name(name)
#             return exp.experiment_id if exp else (
#                 client.create_experiment(name) if allow_create else "0"
#             )
#         except Exception as e:
#             print(f"[DualLogger] ⚠️ Fallback to default experiment due to error: {e}")
#             return "0"
# 
#     def log_metric(self, key, value, step=None):
#         self.local_client.log_metric(self.local_run_id, key, value, step=step)
#         mlflow.log_metric(key, value, step=step)
# 
#     def log_param(self, key, value):
#         self.local_client.log_param(self.local_run_id, key, value)
#         mlflow.log_param(key, value)
# 
#     def log_artifact(self, local_path, artifact_path=None):
#         # Always log artifact to the local backend
#         self.local_client.log_artifact(self.run_id, local_path, artifact_path)
# 
#         # Log to remote backend only if file size is below threshold
#         file_size = os.path.getsize(local_path)
#         if file_size <= self.artifact_size_limit_bytes:
#             self.remote_client.log_artifact(self.run_id, local_path, artifact_path)
#         else:
#             print(f"[DualLogger] ⏭️ Skipped remote logging: {local_path} ({file_size/1024**2:.2f} MB)")
# 
#     def set_tag(self, key, value):
#         self.local_client.set_tag(self.local_run_id, key, value)
#         mlflow.set_tag(key, value)
# 
#     def get_run_id(self):
#         return self.local_run_id
# 
#     def get_shared_run_id(self):
#         return self.shared_run_id
# 
#     def log_batch(self, metrics: dict[str, list[tuple[int, float]]]):
#         timestamp = int(time.time() * 1000)
#         local_metric_objs = []
#         remote_metric_dict = {}
# 
#         for key, values in metrics.items():
#             for step, value in values:
#                 local_metric_objs.append(Metric(key=key, value=value, step=step, timestamp=timestamp))
#                 if step not in remote_metric_dict:
#                     remote_metric_dict[step] = {}
#                 remote_metric_dict[step][key] = value
# 
#         # Efficient batch logging for local
#         self.local_client.log_batch(run_id=self.local_run_id, metrics=local_metric_objs)
# 
#         # Efficient grouped logging for remote (mlflow)
#         for step, metric_dict in remote_metric_dict.items():
#             print("KAKAKA")
#             mlflow.log_metrics(metric_dict, step=step)