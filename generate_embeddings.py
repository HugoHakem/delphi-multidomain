from model import Delphi
from utils import DelphiData, get_batch
from cv_utils import DATA_TYPE_CONFIGS, get_best_ckpt_from_mlflow
import mlflow
import numpy as np
import torch
import argparse
import sys
import os
import json

def filter_and_shift(ids, ages, age_threshold, pad_id=0, healthy_token_id=1, pad_age=-10000):
    """
    Given a 2D tensor of token ids and a 2D tensor of ages (both of shape [batch, seq_len]),
    for each row (subject), remove all tokens (and ages) where age > age_threshold,
    append an EOS token (id=1), right-shift the sequence,
    pad with zeros (for ids) and a constant negative value (for ages) on the left,
    and return the new tensors (same shape as input).

    Args:
        ids (torch.Tensor): 2D tensor of token ids (batch, seq_len).
        ages (torch.Tensor): 2D tensor of ages (batch, seq_len).
        age_threshold (float): Age threshold.
        pad_id (int): Token id for padding (default 0).
        healthy_token_id (int): Token id for EOS/healthy (default 1).
        pad_age (float): Padding value for ages (default -27.3973).

    Returns:
        ids_final (torch.Tensor): 2D tensor, same shape as ids.
        ages_final (torch.Tensor): 2D tensor, same shape as ages.
    """
    batch_size, seq_len = ids.shape
    ids_final = []
    ages_final = []
    for i in range(batch_size):
        ids_row = ids[i]
        ages_row = ages[i]
        # Filter by age
        mask = ages_row <= age_threshold
        ids_filtered = ids_row[mask]
        ages_filtered = ages_row[mask]
        # Append EOS token
        ids_filtered = torch.cat([ids_filtered, torch.tensor([healthy_token_id], device=ids.device, dtype=ids.dtype)])
        ages_filtered = torch.cat([ages_filtered, torch.tensor([age_threshold], device=ages.device, dtype=ages.dtype)])
        # Calculate needed padding
        length = seq_len
        n_pad = length - ids_filtered.shape[0]
        if n_pad < 0:
            # If the filtered sequence + EOS is longer than seq_len, truncate
            ids_filtered = ids_filtered[:length]
            ages_filtered = ages_filtered[:length]
            n_pad = 0
        # Right-shift with padding
        ids_row_final = torch.cat([torch.full((n_pad,), pad_id, dtype=ids.dtype, device=ids.device), ids_filtered])
        ages_row_final = torch.cat([torch.full((n_pad,), pad_age, dtype=ages.dtype, device=ages.device), ages_filtered])
        ids_final.append(ids_row_final)
        ages_final.append(ages_row_final)
    ids_final = torch.stack(ids_final, dim=0)
    ages_final = torch.stack(ages_final, dim=0)
    return ids_final, ages_final


class DelphiEmbeddingInterface:
    def __init__(self, run_id, device=None, dtype='float32', load='all'):
        """
        Interface to load a Delphi model and its associated data from an MLflow run_id.

        Args:
            run_id (str): MLflow run ID.
            device (str): Device to load the model on ('cuda' or 'cpu').
            dtype (str): Data type for tensors.
            load (str): Which data to load: 'all', 'train', or 'val'.
                        'all' loads both, 'train' only training, 'val' only validation.
        """
        
        self.run_id = run_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        # Get run information from MLflow
        runinfo = mlflow.get_run(run_id=run_id)
        data_type = runinfo.data.params['data_type']
        self.config = DATA_TYPE_CONFIGS[data_type]

        # Get the fold used in this run
        if 'fold' in runinfo.data.params:
            val_fold = int(runinfo.data.params['fold'])
        elif 'fold' in runinfo.data.tags:
            val_fold = int(runinfo.data.tags['fold'])
        else:
            raise ValueError("The 'fold' parameter was not found in the MLflow run metadata.")

        self.val_fold = val_fold

        # Get the best checkpoint using the utility function
        self.ckpt_path = get_best_ckpt_from_mlflow(run_id)

        # Load the Delphi model
        self.model = Delphi.from_checkpoint(self.ckpt_path, device=self.device).eval()

        # Load DelphiData using the correct val_fold
        self.delphi_data = DelphiData(
            self.config['data_root'],
            val_fold=self.val_fold,
            delphi_labels=self.config['delphi_labels'],
            labels=self.config['labels'],
            device=self.device
        )
        self.delphi_data.get_p2i()
        self.delphi_data.get_id_to_token()

        # Load the data according to the selected option
        self.train_data = None
        self.val_data = None
        self.all_data = None

        if load == 'all':
            self.train_data = self.delphi_data.train_data
            self.val_data = self.delphi_data.val_data
            self.all_data = np.concatenate([self.delphi_data.train_data, self.delphi_data.val_data])
        elif load == 'train':
            self.train_data = self.delphi_data.train_data
        elif load == 'val':
            self.val_data = self.delphi_data.val_data
        else:
            raise ValueError("The 'load' argument must be 'all', 'train', or 'val'.")

    def get_model(self):
        return self.model

    def get_data(self):
        """
        Returns the DelphiData object.
        """
        return self.delphi_data

    def get_config(self):
        return self.config

    def get_ckpt_path(self):
        return self.ckpt_path

    def get_val_fold(self):
        return self.val_fold

if __name__ == "__main__":
    # Argument parser for number of chunks, age threshold, and output file
    parser = argparse.ArgumentParser(description="Generate embeddings for Delphi subjects.")
    parser.add_argument('--max_chunks', type=int, default=None, help='Maximum number of chunks to process (for testing). If not set, process all.')
    parser.add_argument('--age', type=float, default=30, help='Maximum age (in years) to filter subject events (default: 30).')
    parser.add_argument('--output', type=str, required=True, help='Path to the file where the output will be saved (.json, .npy, or .csv supported).')
    args = parser.parse_args()

   
    # Check output extension early and warn if not supported
    supported_exts = [".json", ".npy", ".csv"]
    output_ext = None
    if args.output is not None:
        output_ext = os.path.splitext(args.output)[1].lower()
        if output_ext not in supported_exts:
            print(f"WARNING: Output extension '{output_ext}' is not supported. Only .json, .npy, and .csv are supported. The output will NOT be saved.")
            args.output = None

    # Interactive selection of experiment and run
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    if not experiments:
        print("No MLflow experiments found.")
        sys.exit(1)

    print("Available MLflow experiments:")
    for idx, exp in enumerate(experiments):
        print(f"{idx}: {exp.name} (ID: {exp.experiment_id})")

    # Ask user to select an experiment
    while True:
        try:
            exp_idx = int(input("Select the experiment number: "))
            if 0 <= exp_idx < len(experiments):
                break
            else:
                print("Invalid experiment number. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    selected_experiment = experiments[exp_idx]
    experiment_id = selected_experiment.experiment_id

    # List all runs for the selected experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000
    )

    if not runs:
        print("No finished runs found for this experiment.")
        sys.exit(1)

    # Sort runs by fold (ascending)
    def get_fold(run):
        if 'fold' in run.data.params:
            return int(run.data.params['fold'])
        elif 'fold' in run.data.tags:
            return int(run.data.tags['fold'])
        else:
            return float('inf')  # If no fold, send to the end

    runs_sorted = sorted(runs, key=get_fold)

    print(f"\nAvailable runs for experiment '{selected_experiment.name}' (sorted by fold):")
    for idx, run in enumerate(runs_sorted):
        run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
        fold = run.data.params.get('fold', run.data.tags.get('fold', 'N/A'))
        print(f"{idx}: Run ID: {run.info.run_id}, Name: {run_name}, Fold: {fold}")

    # Ask user to select a run
    while True:
        try:
            run_idx = int(input("Select the run number: "))
            if 0 <= run_idx < len(runs_sorted):
                break
            else:
                print("Invalid run number. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    selected_run = runs_sorted[run_idx]
    run_id = selected_run.info.run_id

    print(f"\nLoading DelphiEmbeddingInterface for run ID: {run_id} ...")
    interface = DelphiEmbeddingInterface(run_id)
    print("Model, data, and config loaded successfully.")

    model = interface.get_model()
    data = interface.get_data()
    config = interface.get_config()

    # Example usage of DelphiData's new API:
    person, y, last_time = data.get_person(0, data_type="val")
    print("Person (val, idx=0):", person)
    print("y:", y)
    print("last_time:", last_time)

    person_train, y_train, last_time_train = data.get_person(0, data_type="train")
    print("Person (train, idx=0):", person_train)
    print("y_train:", y_train)
    print("last_time_train:", last_time_train)

    person_all, y_all, last_time_all = data.get_person(0, data_type="all")
    print("Person (all, idx=0):", person_all)
    print("y_all:", y_all)
    print("last_time_all:", last_time_all)

    batch_size = 256
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    no_event_token_rate = 5

    ix = torch.randint(len(data.train_p2i), (batch_size,))
    X, A, Y, B = get_batch(ix, data.train_data, data.train_p2i, block_size=block_size, device=device,
                           padding='random', lifestyle_augmentations=True, select='left',
                           no_event_token_rate=no_event_token_rate)
    
    # Split the training dataset into batches of size batch_size
    def chunk_dataset(dataset, batch_size):
        """
        Splits the dataset into chunks (batches) of size batch_size.
        Returns a list of numpy arrays, each with up to batch_size rows.
        """
        return [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

    train_chunks = chunk_dataset(data.train_data, batch_size)
    print(f"The training dataset has been split into {len(train_chunks)} chunks of up to {batch_size} samples each.")

    # We will store a list of dicts: {"subject_id": ..., "embedding": ...}
    embeddings_per_subject = []

    # Determine how many chunks to process
    num_chunks_to_process = len(train_chunks)
    if args.max_chunks is not None:
        num_chunks_to_process = min(args.max_chunks, len(train_chunks))
        print(f"Processing only {num_chunks_to_process} chunk(s) out of {len(train_chunks)} due to --max_chunks argument.")

    # Map integer index to subject_id using the first column of data.train_data
    index_to_subject_id = {i: data.train_data[i][0] for i in range(len(data.train_data))}

    # Filter and shift with the age chosen by the user
    age_threshold_days = args.age * 365.25
    print(f"Filtering events up to age {args.age} years ({age_threshold_days} days).")

    # NOTE: Filtering and shifting must be done per batch, not on global X/A, so that each batch is aligned with the chunk's subjects
    for i, chunk in enumerate(train_chunks):
        if i >= num_chunks_to_process:
            break
        # Get the indices of the subjects in this chunk
        indices = np.arange(i * batch_size, min((i + 1) * batch_size, len(data.train_data)))
        # Prepare the batch using get_batch
        ix = torch.tensor(indices, dtype=torch.long)
        X, A, Y, B, subject_ids = get_batch(ix, data.train_data, data.train_p2i, block_size=block_size, device=device,
                               padding='random', lifestyle_augmentations=True, select='left',
                               no_event_token_rate=no_event_token_rate, return_subject_ids=True)
        # Filter and shift for the chosen age
        X_capped, A_capped = filter_and_shift(X, A, age_threshold=age_threshold_days)
        # Pass the batch through the model to get the embeddings
        with torch.no_grad():
            # Assume the model returns embeddings in the first position
            embeddings = model.get_embeddings(X_capped, A_capped) if hasattr(model, "get_embeddings") else model(X_capped, A_capped, Y=None, B=None)[-1]
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            # For each embedding in the batch, store the subject id and the embedding
            for idx_in_batch, idx_in_data in enumerate(indices):
                subject_id = subject_ids[idx_in_batch].item()
                if subject_id is None:
                    subject_id = f"unknown_{idx_in_data}"
                embeddings_per_subject.append({
                    "subject_id": subject_id,
                    "embedding": embeddings_np[idx_in_batch, -1]
                })
        print(f"Processed chunk {i+1}/{num_chunks_to_process}")

    print(f"Generated embeddings for {len(embeddings_per_subject)} subjects.")

    # Save the output if specified and supported
    output_path = args.output
    ext = os.path.splitext(output_path)[1].lower()
    print(f"Saving output to {output_path} ...")
    for d in embeddings_per_subject:
        if isinstance(d["embedding"], np.ndarray):
            d["embedding"] = [d['subject_id']] + d["embedding"].tolist()
    if ext == ".json":
        # Convert numpy arrays to lists for JSON serialization            
        with open(output_path, "w") as f:
            json.dump(embeddings_per_subject, f)
        print(f"Output saved as JSON to {output_path}")
    elif ext == ".npy":
        np.save(output_path, embeddings_per_subject)
    elif ext == ".csv":
        import pandas as pd
        df = pd.DataFrame([ x['embedding'] for x in embeddings_per_subject ], columns=['subject_id'] + [f'embedding_{str(i).zfill(3)}' for i in range(len(embeddings_per_subject[0]['embedding'])-1)])
        df = df.set_index('subject_id', inplace=False)
        # Format the embedding columns as strings with up to 5 decimals, without trailing zeros
        def format_embedding(emb):
            if isinstance(emb, list):
                return [('{0:.5f}'.format(x).rstrip('0').rstrip('.') if '.' in '{0:.5f}'.format(x) else '{0:.5f}'.format(x)) for x in emb]
            return emb
        for i in range(len(embeddings_per_subject[0]['embedding'])-1):
            df[f'embedding_{str(i).zfill(3)}'] = df[f'embedding_{str(i).zfill(3)}'].apply(format_embedding)
        df.to_csv(output_path, index=False)
        print(f"Output saved as numpy .npy to {output_path}")