from pathlib import Path

mlruns_root = Path("mlruns")

# Paso 1: Listar experimentos
experiments = [p for p in mlruns_root.iterdir() if p.is_dir() and p.name.isdigit()]
experiments.sort(key=lambda p: int(p.name))

print("\nExperimentos encontrados:")
for i, exp in enumerate(experiments, 1):
    print(f"{i}: ID {exp.name}")

# Paso 2: Elegir experimento
try:
    exp_choice = int(input("\nSeleccioná el número del experimento: "))
    selected_exp = experiments[exp_choice - 1]
except (ValueError, IndexError):
    print("Selección inválida.")
    exit()

print(f"\nUsando experimento ID: {selected_exp.name}")

# Paso 3: Listar runs
run_dirs = [d for d in selected_exp.iterdir() if d.is_dir()]
run_info = []

print("\nRuns:")
for i, run_dir in enumerate(run_dirs, 1):
    metrics_file = run_dir / "metrics" / "instant_learning_rate"
    if metrics_file.exists():
        n_lines = sum(1 for _ in metrics_file.open())
        print(f"{i}: {run_dir.name} — {n_lines} líneas")
    else:
        n_lines = 0
        print(f"{i}: {run_dir.name} — ❌ sin archivo 'instant_learning_rate'")
    run_info.append((run_dir.name, metrics_file, n_lines))

# Paso 4: Elegir run
try:
    run_choice = int(input("\nElegí el número de la run (0 para TODAS): "))
except ValueError:
    print("Selección inválida.")
    exit()

def procesar_run(run_id, path):
    with path.open("r") as f:
        original_lines = f.readlines()

    filtered = [
        line.strip() for line in original_lines
        if int(line.strip().split()[-1]) % 100 == 0
    ]

    seen = set()
    filtered_unique = []
    for line in filtered:
        if line not in seen:
            filtered_unique.append(line)
            seen.add(line)

    filtered_lines = [line + "\n" for line in filtered_unique]
    return original_lines, filtered_lines

if run_choice == 0:
    # Proceso en batch todas las runs con archivo válido
    runs_to_process = [(run_id, path) for run_id, path, n_lines in run_info if path.exists()]

    print(f"\nVas a procesar {len(runs_to_process)} runs.")
    confirm = input("¿Proceder con el reemplazo de todos los archivos? (y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Operación cancelada.")
        exit()

    for run_id, path in runs_to_process:
        orig, filtered = procesar_run(run_id, path)
        with path.open("w") as f:
            f.writelines(filtered)
        print(f"✅ {run_id}: {len(orig)} → {len(filtered)} líneas")
    print("\n🏁 Limpieza completada.")
else:
    try:
        run_id, path, total_lines = run_info[run_choice - 1]
    except IndexError:
        print("Selección inválida.")
        exit()

    if not path.exists():
        print(f"\n❌ El archivo {path} no existe.")
        exit()

    orig, filtered = procesar_run(run_id, path)

    print(f"\nRun seleccionada: {run_id}")
    print(f"Líneas originales: {len(orig)}")
    print(f"Líneas luego del filtrado y deduplicado: {len(filtered)}\n")

    print("Primeras líneas del nuevo archivo:")
    for line in filtered[:5]:
        print(line.strip())
    if len(filtered) > 10:
        print("...")
    print("\nÚltimas líneas del nuevo archivo:")
    for line in filtered[-5:]:
        print(line.strip())

    confirm = input("\n¿Reemplazar archivo original? (y/N): ").strip().lower()
    if confirm == "y":
        with path.open("w") as f:
            f.writelines(filtered)
        print("✅ Archivo reemplazado.")
    else:
        print("❌ Operación cancelada.")

