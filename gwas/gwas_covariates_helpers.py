import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO)


def ukb_gen_read_sample(file, col_names=("id_1", "id_2", "missing"), row_skip=2):
    """Read UKB sample file (equivalent to ukbtools)."""
    df = pd.read_csv(file, delim_whitespace=True, skiprows=row_skip, names=col_names)
    return df


def generate_covariates_df(covariates_config, impute_with_mean_for=None):
    """
    Load covariates from a config dict (parsed from YAML).
    Columns can be specified as strings or as {original_name: new_name}.
    
    Example YAML:
    data/covariates.csv:
      - id: "eid"
      - X50: Height
      - X4079: DBP
    data/genomicPCs_unrelated_GBR.tsv:
      - id: "ID"
      - PC1
      - PC2
    """
    covariates_df = None
    covariate_names = []

    for covfile, spec in covariates_config.items():
        delim = _infer_delim(covfile)
        df_ = pd.read_csv(covfile, sep=delim)

        id_colname = spec[0]["id"]
        df_ = df_.rename(columns={id_colname: "ID"})
        df_["ID"] = df_["ID"].astype(str)

        rename_map = {}
        new_covariates = []

        for entry in spec[1:]:
            if isinstance(entry, str):
                new_covariates.append(entry)
            elif isinstance(entry, dict):
                # YAML like {X50: Height}
                orig, new = list(entry.items())[0]
                rename_map[orig] = new
                new_covariates.append(new)
            else:
                raise ValueError(f"Unexpected spec entry: {entry}")

        if rename_map:
            df_ = df_.rename(columns=rename_map)

        covariate_names.extend(new_covariates)

        if covariates_df is None:
            covariates_df = df_
        else:
            covariates_df = covariates_df.merge(df_, on="ID", how="left")

    if impute_with_mean_for is not None:
        covariates_df = impute_na(covariates_df, impute_with_mean_for)

    before = covariates_df.shape[0]
    covariates_df = covariates_df.dropna()
    after = covariates_df.shape[0]
    if before != after:
        logging.warning(f"Dropped {before - after} rows due to missing covariates.")

    covariates_df = covariates_df[["ID"] + covariate_names]
    return covariates_df


def impute_na(df, columns, method="mean"):
    """Impute missing values with column means."""
    columns = [c for c in columns if c in df.columns]
    for col in columns:
        if method == "mean":
            mean_val = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_val)
    return df


def read_raw_pheno(pheno_file, pheno_names=None, exclude_columns=None):
    """Read phenotype file into DataFrame."""
    if not Path(pheno_file).exists():
        logging.error(f"File {pheno_file} not found.")
        raise FileNotFoundError(pheno_file)

    delim = _infer_delim(pheno_file)
    pheno_df = pd.read_csv(pheno_file, sep=delim)

    if "ID" not in pheno_df.columns:
        logging.error("Column ID missing in phenotype file.")
        raise ValueError("Missing ID column")

    pheno_df.index = pheno_df["ID"].astype(str)

    if pheno_names is None:
        pheno_names = list(pheno_df.columns)
        if exclude_columns is not None:
            pheno_names = [c for c in pheno_names if c not in exclude_columns]

    return pheno_df[pheno_names]


def fit_linear_model(df, phenotype, covariates):
    """Fit linear model phenotype ~ covariates."""
    X = df[covariates]
    X = sm.add_constant(X)
    y = df[phenotype]
    model = sm.OLS(y, X, missing="drop")
    fit = model.fit()
    return fit


def inverse_normalise(x):
    """Rank-based inverse normalisation."""
    ranks = x.rank(method="average", na_option="keep")
    normed = (ranks - 0.5) / ranks.notna().sum()
    return pd.Series(normed).apply(lambda p: sm.distributions.norm.ppf(p))


def adj_by_covariates(raw_pheno_df, covariates_df):
    """Adjust phenotypes by covariates using linear regression."""
    pheno_names = [c for c in raw_pheno_df.columns if c != "ID"]
    covariate_names = [c for c in covariates_df.columns if c != "ID"]

    adj_pheno_df = raw_pheno_df.copy()
    merged = raw_pheno_df.merge(covariates_df, on="ID", how="left")

    fit_summaries = {}

    for pheno in pheno_names:
        logging.info(f"Processing phenotype {pheno}...")
        fit = fit_linear_model(merged, pheno, covariate_names)

        adj_resid = fit.resid
        adj_norm = inverse_normalise(pd.Series(adj_resid))
        adj_pheno_df[pheno] = adj_norm

        fit_summaries[pheno] = fit.summary().tables[1]  # coefficients

    return {"adj_pheno_df": adj_pheno_df, "fit_summaries": fit_summaries}


def format_df_for_tool(pheno_df, gwas_software="plink", ukb_sample=None):
    """Format phenotype DataFrame for PLINK or BGENIE."""
    pheno_names = [c for c in pheno_df.columns if c != "ID"]
    gwas_software = gwas_software.lower()

    if gwas_software == "plink":
        logging.info("Formatting table for Plink...")
        pheno_df = pheno_df.rename(columns={"ID": "IID"})
        pheno_df["FID"] = pheno_df["IID"]
        pheno_df = pheno_df[["FID", "IID"] + pheno_names]

    elif gwas_software == "bgenie":
        logging.info("Formatting table for BGENIE...")
        if ukb_sample is None:
            raise ValueError("Sample file required for BGENIE")

        sample_df = ukb_gen_read_sample(ukb_sample)
        sample_df = sample_df.rename(columns={sample_df.columns[0]: "ID"})
        sample_df["ID"] = sample_df["ID"].astype(str)

        logging.info("Ordering table according to BGEN samples file...")
        pheno_df = sample_df.merge(pheno_df, on="ID", how="left")[["ID"] + pheno_names]

    return pheno_df


def _infer_delim(path):
    """Guess delimiter from file extension."""
    if str(path).endswith(".tsv") or str(path).endswith("txt"):
        return "\t"
    return ","  # default
