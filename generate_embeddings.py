from model import Delphi
from utils import DelphiData, get_batch
from cv_utils import DATA_TYPE_CONFIGS, get_best_ckpt_from_mlflow
import mlflow
import numpy as np
import torch

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
    # Interactive selection of experiment and run
    import sys

    # List all available experiments
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
    # You can now use interface.get_model(), interface.get_data(), etc.
    
    model = interface.get_model()
    data = interface.get_data()
    config = interface.get_config()
    # ckpt_path = interface.get_ckpt_path()
    # val_fold = interface.get_val_fold()

    # print(model)
    # print(data)
    # print(config)
    # print(ckpt_path)

    # Example usage of DelphiData's new API:
    # Get a person from validation set (index 0)
    person, y, last_time = data.get_person(0, data_type="val")
    print("Person (val, idx=0):", person)
    print("y:", y)
    print("last_time:", last_time)

    # Get a person from training set (index 0)
    person_train, y_train, last_time_train = data.get_person(0, data_type="train")
    print("Person (train, idx=0):", person_train)
    print("y_train:", y_train)
    print("last_time_train:", last_time_train)

    # Get a person from all data (index 0)
    person_all, y_all, last_time_all = data.get_person(0, data_type="all")
    print("Person (all, idx=0):", person_all)
    print("y_all:", y_all)
    print("last_time_all:", last_time_all)

    batch_size = 128
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    no_event_token_rate = 5

    ix = torch.randint(len(data.train_p2i), (batch_size,))
    X, A, Y, B = get_batch(ix, data.train_data, data.train_p2i, block_size=block_size, device=device,
                           padding='random', lifestyle_augmentations=True, select='left',
                           no_event_token_rate=no_event_token_rate)
    
    # Chunk the training dataset into batches of size batch_size
    def chunk_dataset(dataset, batch_size):
        """
        Splits the dataset into chunks (batches) of size batch_size.
        Returns a list of numpy arrays, each with up to batch_size rows.
        """
        return [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

    # Example usage with the training data
    train_chunks = chunk_dataset(data.train_data, batch_size)
    print(f"The training dataset has been split into {len(train_chunks)} chunks of up to {batch_size} samples each.")
    # The idea is to iterate over each chunk (batch) of the training dataset and generate the output/embedding for each subject.
    # Here we assume that the model and get_batch function are properly defined and that the model returns embeddings or logits.
    # We will store the embeddings for each subject in a list.

    embeddings_per_subject = []

    for i, chunk in enumerate(train_chunks):
        # Get the indices of the subjects in this chunk
        indices = np.arange(i * batch_size, min((i + 1) * batch_size, len(data.train_data)))
        # Prepare the batch using get_batch
        ix = torch.tensor(indices, dtype=torch.long)
        X, A, Y, B = get_batch(ix, data.train_data, data.train_p2i, block_size=block_size, device=device,
                               padding='random', lifestyle_augmentations=True, select='left',
                               no_event_token_rate=no_event_token_rate)
        # Pass the batch through the model to get the embeddings
        with torch.no_grad():
            # Assume the model returns embeddings in the first position
            embeddings = model.get_embeddings(X, A, Y, B) if hasattr(model, "get_embeddings") else model(X, A, Y, B)[0]
            # Convert to numpy and store
            embeddings_np = embeddings.cpu().numpy()
            embeddings_per_subject.extend(embeddings_np)
        print(f"Processed chunk {i+1}/{len(train_chunks)}")

    print(f"Generated embeddings for {len(embeddings_per_subject)} subjects.")
