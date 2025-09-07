# %%
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def load_one_field(path):
    with open(path) as f:
        data = json.load(f)

    two_field_to_pseudo = defaultdict(list)

    for pseudoseq, entry in data.items():
        locus = entry["canonical_allele"]["locus"]
        prefixed_seq = f"{locus}:{pseudoseq}"
        for allele_info in entry["alleles"]:
            allele_full = allele_info["gene_allele_name"]

            # Reduce to one field
            gene, full_fields = allele_full.split("*")
            fields = full_fields.split(":")
            if len(fields) >= 2:
                two_field = f"{gene}*{fields[0]}:{fields[1]}"
                two_field_to_pseudo[two_field].append(prefixed_seq)

    # Solve frequency conflicts 
    final_map = {}
    for allele, pseudo_list in two_field_to_pseudo.items():
        count = Counter(pseudo_list)
        most_common, freq = count.most_common(1)[0]
        if len(count) > 1:
            print(f"Ambigüedad para {allele}: {dict(count)} → se toma '{most_common}' (más frecuente)")
        final_map[allele] = most_common

    return final_map

map_a = load_one_field("data/pseudosequences_hla_a.json")
map_b = load_one_field("data/pseudosequences_hla_a.json")
map_c = load_one_field("data/pseudosequences_hla_c.json")
allele_to_pseudosequence = map_a | map_b | map_c

delphi_labels = pd.read_csv("delphi_labels_chapters_colours_icd_with_hla4d.csv")
possible_alleles = set([ x for x in delphi_labels.name.iloc[4:4+359].to_list() ])

inverse_dict = defaultdict(list)
for allele, pseudo in allele_to_pseudosequence.items():
    if allele in possible_alleles:
        inverse_dict[pseudo].append(allele)
inverse_dict = dict(inverse_dict)

os.makedirs("data/ukb_real_5_folds_peptide_pseudoseq/")

vocab = pd.read_csv(f"data/ukb_real_data_4digit/labels.csv", header=None)
vocab = vocab.reset_index().rename({"index": "id", 0: "token"}, axis=1)

# 1. Load vocabulary
# 2. Keep only rows that are HLA alleles
hla_vocab = vocab[vocab["id"].between(4, 362)]  # ajustá si cambia el rango
possible_alleles = set([ x for x in vocab.token.to_list() if x.startswith("HLA") ])
allele_to_pseudosequence = { k: v for k, v in allele_to_pseudosequence.items() if k in possible_alleles }

# %%
# 3. Pseudo-sequence -> list of tokens
pseudoseq_to_tokens = defaultdict(list)
for _, row in hla_vocab.iterrows():
    token = row["token"]
    if token in allele_to_pseudosequence:
        pseudoseq = allele_to_pseudosequence[token]
        pseudoseq_to_tokens[pseudoseq].append(token)

token_to_id = dict(zip(vocab["token"], vocab["id"]))
collapsed_groups = {}
for new_id, tokens in enumerate(pseudoseq_to_tokens.values()):
    original_ids = [token_to_id[tok] for tok in tokens]
    collapsed_groups[new_id] = original_ids


# %%
for i in range(1,6):
    data = np.memmap(f"data/ukb_real_5_folds_4digit/fold{i}.bin", dtype=np.int32).reshape(-1,3)

allele_to_pseudosequence
sorted(inverse_dict.values())
len(set(list(map_a.values())))
