import mlflow

mlflow.set_tracking_uri("https://dagshub.com/rbonazzola/delphi.mlflow")

client = mlflow.tracking.MlflowClient()
print(dir(client))
client.search_experiments()
