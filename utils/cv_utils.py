import mlflow
import re
import os

DATA_TYPE_CONFIGS = {

    'real-hla-4digits-5folds': {
        'labels': './data/ukb_real_5_folds_4digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla4d.csv',
        'data_root': './data/ukb_real_5_folds_4digit',
    },
    'real-nohla-5folds': {
        'labels': './data/ukb_real_5_folds_nohla/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd.csv',
        'data_root': './data/ukb_real_5_folds_nohla',
    },
    'real-hla-2digits-5folds': {
        'labels': './data/ukb_real_5_folds_2digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla2d.csv',
        'data_root': './data/ukb_real_5_folds_2digit',
    },
}


def get_best_ckpt_from_mlflow(runid):
    runinfo = mlflow.get_run(run_id=runid)
    ckpt_dir = re.sub(r'^.*(?=mlruns)', '', runinfo.info.artifact_uri) + '/checkpoints'
    best_ckpt = [x for x in os.listdir(ckpt_dir) if 'best' in x][0]
    best_ckpt = os.path.join(ckpt_dir, best_ckpt)
    return best_ckpt

def get_run_from_fold(experiment_id=None, val_fold=None):
    """ 
    Retrieves val_fold and the checkpoint path from MLflow metadata using experiment_id.
    Ensures the experiment contains all 5 folds.
    Returns (val_fold, run_id, ckpt_path).
    """

    experiment_id = str(experiment_id)

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000
    )   
    if not runs:
        raise ValueError(f"No successful runs found in experiment {experiment_id}")

    val_folds = []
    fold_to_run = {}
    for run in runs:
        val_fold_run = None
        if 'fold' in run.data.params:
            val_fold_run = int(run.data.params['fold'])
        elif 'fold' in run.data.tags:
            val_fold_run = int(run.data.tags['fold'])
        if val_fold_run is not None:
            if val_fold_run not in fold_to_run:
                fold_to_run[val_fold_run] = run 
                val_folds.append(val_fold_run)

    if sorted(val_folds) != [1, 2, 3, 4, 5]: 
        raise ValueError(f"Experiment {experiment_id} does not contain all 5 folds! Found folds: {sorted(val_folds)}")

    requested_fold = val_fold

    if requested_fold not in fold_to_run:
        raise ValueError(f"Requested fold {requested_fold} not found in experiment {experiment_id}")

    run = fold_to_run[requested_fold]

    return run.info.run_id
