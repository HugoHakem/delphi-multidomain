import mlflow

import mlflow
import re

THRESHOLD = 12.
OUTPUT_FILE = "auc_runs.txt"

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

lines = []

for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        run_id = run.info.run_id
        metrics = run.data.metrics
        if "best_val_loss" in metrics:
            val = metrics["best_val_loss"]
            if val < THRESHOLD:
                # Buscar checkpoint de mayor step
                ckpt_dir = "checkpoints"
                artifacts = client.list_artifacts(run_id, path=ckpt_dir)
                candidates = []
                for a in artifacts:
                    match = re.search(rf"best_ckpt__{run_id}__(\d+)\.pt", a.path)
                    if match:
                        step = int(match.group(1))
                        candidates.append(step)
                if not candidates:
                    print(f"⚠️  Run {run_id} no tiene checkpoints válidos.")
                    continue
                best_step = max(candidates)
                lines.append(f"{exp.experiment_id} {run_id} {val:.6f} {best_step}")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"✅ Archivo '{OUTPUT_FILE}' generado con {len(lines)} entradas.")
