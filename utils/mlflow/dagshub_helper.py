import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

from mlflow.entities import Metric

def sync_run_to_dagshub(local_uri, remote_uri, local_run_id, artifact_size_limit_mb=10):
    from mlflow.tracking import MlflowClient
    import mlflow
    import os
    import dagshub
    import time

    local_client = MlflowClient(tracking_uri=local_uri)
    remote_client = MlflowClient(tracking_uri=remote_uri)

    # Get local run
    local_run = local_client.get_run(local_run_id)
    experiment_name = local_client.get_experiment(local_run.info.experiment_id).name

    # Init DagsHub repo and set tracking URI
    dagshub.init(repo_name=experiment_name, repo_owner="rbonazzola", mlflow=True)
    mlflow.set_tracking_uri(remote_uri)

    # Create run in DagsHub
    with mlflow.start_run(run_name=local_run.data.tags.get("mlflow.runName", None)) as remote_run:
        remote_run_id = remote_run.info.run_id

        # Tags
        for k, v in local_run.data.tags.items():
            mlflow.set_tag(k, v)

        # Params
        for k, v in local_run.data.params.items():
            mlflow.log_param(k, v)

        # Metrics (batch logging)
        timestamp = int(time.time() * 1000)
        metric_objs = []
        for metric_name in local_run.data.metrics.keys():
            try:
                history = local_client.get_metric_history(local_run_id, metric_name)
                for m in history:
                    metric_objs.append(Metric(key=metric_name, value=m.value, step=m.step, timestamp=m.timestamp or timestamp))
            except Exception as e:
                print(f"[sync] ⚠️ Could not load history for metric {metric_name}: {e}")
        if metric_objs:
            mlflow.tracking.MlflowClient().log_batch(remote_run_id, metrics=metric_objs)

        # Artifacts
        local_artifacts_path = local_client.download_artifacts(local_run_id, "")
        for root, _, files in os.walk(local_artifacts_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, local_artifacts_path)
                size = os.path.getsize(fpath)
                if size <= artifact_size_limit_mb * 1024 * 1024:
                    mlflow.log_artifact(fpath, os.path.dirname(relpath))
                else:
                    print(f"[sync] ⏭️ Skipping large artifact: {relpath} ({size / 1024 ** 2:.2f} MB)")

        print(f"[sync] ✅ Run synced to DagsHub: {remote_run_id}")


# def sync_run_to_dagshub(local_uri, remote_uri, local_run_id, artifact_size_limit_mb=10):
#     from mlflow.tracking import MlflowClient
#     import mlflow
#     import os
#     import dagshub
# 
#     local_client = MlflowClient(tracking_uri=local_uri)
#     remote_client = MlflowClient(tracking_uri=remote_uri)
# 
#     # Get local run
#     local_run = local_client.get_run(local_run_id)
#     experiment_name = local_client.get_experiment(local_run.info.experiment_id).name
# 
#     # Init DagsHub repo
#     dagshub.init(repo_name=experiment_name, repo_owner="rbonazzola", mlflow=True)
#     mlflow.set_tracking_uri(remote_uri)
# 
#     # Create run in DagsHub
#     with mlflow.start_run(run_name=local_run.data.tags.get("mlflow.runName", None)) as remote_run:
#         remote_run_id = remote_run.info.run_id
# 
#         # Copy tags
#         for k, v in local_run.data.tags.items():
#             mlflow.set_tag(k, v)
# 
#         # Copy params
#         for k, v in local_run.data.params.items():
#             mlflow.log_param(k, v)
# 
#         # Copy metrics
#         for metric_name in local_run.data.metrics.keys():
#             try:
#                 for m in local_client.get_metric_history(local_run_id, metric_name):
#                     mlflow.log_metric(metric_name, m.value, step=m.step, timestamp=m.timestamp)
#             except Exception as e:
#                 print(f"[sync] ⚠️ Could not log metric {metric_name}: {e}")
# 
#         # Copy artifacts
#         local_artifacts_path = local_client.download_artifacts(local_run_id, "")
#         for root, _, files in os.walk(local_artifacts_path):
#             for fname in files:
#                 fpath = os.path.join(root, fname)
#                 relpath = os.path.relpath(fpath, local_artifacts_path)
#                 size = os.path.getsize(fpath)
#                 if size <= artifact_size_limit_mb * 1024 * 1024:
#                     mlflow.log_artifact(fpath, os.path.dirname(relpath))
#                 else:
#                     print(f"[sync] ⏭️ Skipping large artifact: {relpath} ({size / 1024 ** 2:.2f} MB)")
# 
#         print(f"[sync] ✅ Run synced to DagsHub: {remote_run_id}")


# def sync_run_to_dagshub(local_uri, remote_uri, local_run_id, artifact_size_limit_mb=10):
#     local_client = MlflowClient(tracking_uri=local_uri)
#     remote_client = MlflowClient(tracking_uri=remote_uri)
# 
#     # Get local run
#     local_run = local_client.get_run(local_run_id)
#     experiment_name = local_client.get_experiment(local_run.info.experiment_id).name
# 
#     # Init DagsHub repo
#     import dagshub
#     dagshub.init(repo_name=experiment_name, repo_owner="rbonazzola", mlflow=True)
# 
#     mlflow.set_tracking_uri(remote_uri)
#     
#     # Create run in DagsHub
#     with mlflow.start_run(run_name=local_run.data.tags.get("mlflow.runName", None)) as remote_run:
#         remote_run_id = remote_run.info.run_id
# 
#         # Copy tags
#         for k, v in local_run.data.tags.items():
#             mlflow.set_tag(k, v)
# 
#         # Copy params
#         for k, v in local_run.data.params.items():
#             mlflow.log_param(k, v)
# 
#         # Copy metrics
#         metrics = local_client.get_metric_history(local_run_id, local_run.data.metrics.keys())
#         for metric_name in local_run.data.metrics.keys():
#             for m in local_client.get_metric_history(local_run_id, metric_name):
#                 mlflow.log_metric(metric_name, m.value, step=m.step, timestamp=m.timestamp)
# 
#         # Copy artifacts
#         local_artifacts_path = local_client.download_artifacts(local_run_id, "")
#         for root, _, files in os.walk(local_artifacts_path):
#             for fname in files:
#                 fpath = os.path.join(root, fname)
#                 relpath = os.path.relpath(fpath, local_artifacts_path)
#                 size = os.path.getsize(fpath)
#                 if size <= artifact_size_limit_mb * 1024 * 1024:
#                     mlflow.log_artifact(fpath, os.path.dirname(relpath))
#                 else:
#                     print(f"[sync] ⏭️ Skipping large artifact: {relpath} ({size / 1024 ** 2:.2f} MB)")
# 
#         print(f"[sync] ✅ Run synced to DagsHub: {remote_run_id}")
# 
