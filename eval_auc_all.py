import mlflow
import re
from pathlib import Path

# Parámetros
BEST_VAL_LOSS_THRESHOLD = 12.0
OUTPUT_FILENAME = ""

client = mlflow.tracking.MlflowClient()

# Paso 1: Listar experimentos activos
experiments = client.search_experiments()
print("\nExperimentos disponibles:")
for i, exp in enumerate(experiments, start=1):
    print(f"{i}: {exp.name} (ID: {exp.experiment_id})")

# Paso 2: Selección de experimento
try:
    choice = int(input("\nSeleccioná el número del experimento: "))
    selected_experiment = experiments[choice - 1]
except (ValueError, IndexError):
    print("Selección inválida.")
    exit()

print(f"\nUsando experimento: {selected_experiment.name} (ID: {selected_experiment.experiment_id})")

# Paso 3: Buscar corridas válidas
runs = client.search_runs(
    experiment_ids=[selected_experiment.experiment_id],
    order_by=["attributes.start_time DESC"]
)

for run in runs:
    run_id = run.info.run_id
    metrics = run.data.metrics

    if "best_val_loss" not in metrics:
        print(f"❌ Run {run_id} no tiene 'best_val_loss'. Saltando.")
        continue

    best_val_loss = metrics["best_val_loss"]
    if best_val_loss >= BEST_VAL_LOSS_THRESHOLD:
        print(f"⚠️ Run {run_id} tiene best_val_loss={best_val_loss:.4f} >= {BEST_VAL_LOSS_THRESHOLD}. Saltando.")
        continue

    # Paso 4: Buscar checkpoint con mayor step
    checkpoint_dir = "checkpoints"
    artifacts = client.list_artifacts(run_id, path=checkpoint_dir)
    ckpt_files = [f for f in artifacts if f.path.startswith(f"{checkpoint_dir}/best_ckpt__{run_id}__")]

    step_ckpts = []
    for f in ckpt_files:
        match = re.search(rf"best_ckpt__{re.escape(run_id)}__(\d+)\.pt", f.path)
        if match:
            step = int(match.group(1))
            step_ckpts.append((step, f.path))

    if not step_ckpts:
        print(f"❌ No se encontraron checkpoints válidos para run {run_id}.")
        continue

    step_ckpts.sort(reverse=True)
    best_ckpt_path = step_ckpts[0][1]

    # Paso 5: Construir paths
    base_path = Path("mlruns") / selected_experiment.experiment_id / run_id / "artifacts"
    ckpt_path = base_path / best_ckpt_path
    output_path = base_path / "auc" / OUTPUT_FILENAME

    # Paso 6: Construir comando
    command = f"""python evaluate_auc.py \\
  --input_path data/ukb_real_data/ \\
  --data_file_prefix "ukb_real_hla_" \\
  --output_path {output_path} \\
  --model_ckpt_path {ckpt_path} \\
  --no_event_token_rate 5 \\
  --health_token_replacement_prob 0.0 \\
  --dataset_subset_size -1 \\
  --n_bootstrap 100 \\
  --filter_min_total 100 \\
  --disease_chunk_size 200"""

    print(f"\n✅ Run {run_id} (best_val_loss={best_val_loss:.4f}):")
    print(command)
