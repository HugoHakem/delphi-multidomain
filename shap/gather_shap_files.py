import pickle
from tqdm import tqdm 
import glob
import numpy as np
import re

# Find all files matching the pattern
runid = "f945b0c1f1b84294bf29a6d39e6ef831"
files = sorted(glob.glob(f"shap_values_chunk*of*_{runid}.pkl"))

# Function to extract the chunk number from the filename
def extract_chunk_number(filename):
    match = re.search(r"chunk(\d+)of", filename)
    return int(match.group(1)) if match else -1

# Sort files by chunk number
sorted_files = sorted(files, key=extract_chunk_number)

# Initialize lists for each field
all_tokens, all_values, all_times, all_people = [], [], [], []

for file in tqdm(sorted_files):
    print(file)
    with open(file, "rb") as f:
        data = pickle.load(f)
        all_tokens.append(data['tokens'])
        all_values.append(data['values'])
        all_times.append(data['times'])
        all_people.append(data['people'])

# Concatenate arrays/lists
all_tokens = np.concatenate(all_tokens)
all_values = np.concatenate(all_values, axis=1)[0] if all_values[0].ndim == 3 else np.concatenate(all_values)
all_times = np.concatenate(all_times)
all_people = np.concatenate(all_people)

# Save the result
with open(f"shap_values_{runid}_concatenated.pkl", "wb") as f:
    pickle.dump({
        'tokens': all_tokens,
        'values': all_values,
        'times': all_times,
        'people': all_people
    }, f)

print("Done! Concatenated and ordered file: shap_values_concatenated.pkl")
