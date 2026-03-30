# MLflow tips

## Environment

Set the tracking URI so all `mlflow` commands point to the right place:

```bash
export MLFLOW_TRACKING_URI="/path/to/this/repo/train_scripts/mlruns"
```

Add this to your `~/.bashrc` so it is always available.

## Useful aliases

Add these to your `~/.bashrc`:

```bash
# List all experiments
alias mes="mlflow experiments search"

# List runs for a given experiment by its numeric ID
# Usage: mrl <experiment-id>
# Example: mrl 3
mrl() {
    local expid=$(mes | grep -e "^$1" | awk '{print $1}')
    [ -z "$expid" ] && { echo "Usage: mrl <experiment-id>" >&2; return 1; }
    mlflow runs list --experiment-id "$expid" 2>&1
}
```

## MLflow UI

Launch the UI with:

```bash
mlflow ui
```

This starts a local web server (by default at `http://localhost:5000`) that reads from `MLFLOW_TRACKING_URI`.

On **Codon**, the UI runs on the remote host so you need SSH port forwarding to access it from your browser. Connect with:

```bash
ssh -L 5000:localhost:5000 codon.ebi.ac.uk
```

Then start `mlflow ui` on Codon and open `http://localhost:5000` in your local browser. If port 5000 is already taken, pick any free port (e.g. 5001) and use it consistently in both the `ssh` and `mlflow ui --port 5001` commands.

The UI allows you to:
- Rename experiments and runs
- Add or edit tags and notes on runs
- Compare metrics across runs with interactive plots
- Delete runs or archive experiments
- Download artifacts

It does not support moving runs between experiments or bulk editing.

### Typical workflow

```bash
# 1. See all experiments and their IDs
mes

# Experiment ID  Name                    ...
# 1              baseline
# 2              hla_bidir
# 3              rare_variants

# 2. List runs for experiment 3
mrl 3
```
