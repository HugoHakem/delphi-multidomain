import os
import re
from pathlib import Path

root_dir = Path("./mlruns/855723043355657924")

for run_dir in root_dir.iterdir():
    
    checkpoints_dir = run_dir / "artifacts" / "checkpoints"
    
    if not checkpoints_dir.exists():
        continue

    pt_files = list(checkpoints_dir.glob("best_ckpt__*.pt"))
    if not pt_files:
        continue

    def get_step(f):
        match = re.search(r"__(\d+)\.pt$", f.name)
        return int(match.group(1)) if match else -1

    max_file = max(pt_files, key=get_step)
    to_delete = sorted([f for f in pt_files if f != max_file], key=lambda x: int(x.name.split("__")[2].split(".")[0]))

    if to_delete:
        print(f"\nEn carpeta: {checkpoints_dir}")
        print(f"Se va a conservar:\n  {max_file.name}")
        print("Se van a eliminar:")
        for f in to_delete:
            print(f"  {f.name}")
        confirm = input("¿Confirmar eliminación? (y/n): ")
        if confirm.lower() == 'y':
            for f in to_delete:
                f.unlink()
            print("Archivos eliminados.")
        else:
            print("Saltado.")
