import mlflow

import mlflow
import re

METRIC = "best_val_loss"
METRIC = "val_loss"

THRESHOLD = 11.93
OUTPUT_FILE = "auc_runs.txt"

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

lines = []

for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        run_id = run.info.run_id
        metrics = run.data.metrics
        if METRIC in metrics:
            val = metrics[METRIC]
            if val < THRESHOLD:
                # Buscar checkpoint de mayor step
                ckpt_dir = "checkpoints"
                artifacts = client.list_artifacts(run_id, path=ckpt_dir)
                candidates = []
                ckpt_files = []
                for a in artifacts:
                    match = re.search(rf"best_ckpt__{run_id}__(\d+)\.pt", a.path)
                    if match:
                        step = int(match.group(1))
                        ckpt_file = f"best_ckpt__{run_id}__{step}.pt"
                        candidates.append(step)
                        ckpt_files.append(ckpt_file) 
                    else:
                        match = re.search(rf"ckpt__{run_id}__(\d+)\.pt", a.path)
                        if match:
                            step = int(match.group(1))
                            ckpt_file = f"ckpt__{run_id}__{step}.pt"
                            candidates.append(step)
                            ckpt_files.append(ckpt_file) 
                if not candidates:
                    print(f"⚠️  Run {run_id} no tiene checkpoints válidos.")
                    continue
                best_step = max(candidates)
                best_ckpt_file = ckpt_files[candidates.index(best_step)]
                lines.append(f"{exp.experiment_id} {run_id} {val:.6f} {best_step} {best_ckpt_file}")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"✅ Archivo '{OUTPUT_FILE}' generado con {len(lines)} entradas.")
