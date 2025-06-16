import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

st.set_page_config(layout="wide")
st.title("MLflow Run Explorer")

# Sidebar controls
st.sidebar.title("Settings")
mlflow_tracking_uri = st.sidebar.text_input("Tracking URI", value="./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

experiments = client.search_experiments()
exp_name_to_id = {e.name: e.experiment_id for e in experiments}
selected_exp_name = st.sidebar.selectbox("Select Experiment", list(exp_name_to_id.keys()))
experiment_id = exp_name_to_id[selected_exp_name]

# max_results = st.sidebar.slider("Max number of runs", 10, 1000, 200)
val_loss_threshold = st.sidebar.number_input("Filter runs with val_loss less than:", value=12.5, step=0.01)

if "runs_loaded" not in st.session_state:
    st.session_state.runs_loaded = False

if st.sidebar.button("Load runs"):
    st.session_state.runs_loaded = True

@st.cache_data
def load_runs(experiment_ids: str, val_loss_threshold: float = 1.0):

    client = MlflowClient()
    filter_str = f"metrics.val_loss < {val_loss_threshold}"
    print(f"{experiment_ids=}")

    try:
        runs = client.search_runs(
            experiment_ids if type(experiment_ids) == list else [experiment_ids],
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string=filter_str,
        )
    except Exception as e:
        print(f"Error searching for runs: {e}")
        return pd.DataFrame(), 0

    records = []
    skipped = 0

    for run in tqdm(runs):
        try:
            if run.info.status != "FINISHED":
                continue
            row = {
                "run_id": run.info.run_id,
                **run.data.params,
                **run.data.metrics
            }
            records.append(row)
        except Exception as e:
            skipped += 1
            print(f"Skipping run {run.info.run_id}: {e}")

    if not records:
        return pd.DataFrame(), 0

    df = pd.DataFrame(records)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            continue

    df = df.loc[:, df.nunique(dropna=False) > 1]

    return df, skipped


@st.cache_data
def get_loss_curve(run_id, metric="val_loss"):
    try:
        history = client.get_metric_history(run_id, metric)
        return pd.DataFrame({
            "step": [m.step for m in history],
            "value": [m.value for m in history],
            "run_id": run_id
        })
    except:
        return None


# Visualization if runs were loaded
if st.session_state.runs_loaded:

    df, skipped = load_runs(experiment_id, val_loss_threshold)

    if df.empty:
        st.warning("No valid runs could be loaded.")
        st.stop()

    st.success(f"{len(df)} runs loaded. {skipped} discarded.")

    tab1, tab2, tab3, tab4 = st.tabs(["\U0001F4CB Runs", "\U0001F4C8 Correlations", "\U0001F4C9 Loss Curves", "AUC"])

    with tab1:
        st.subheader("Runs Table")
        st.dataframe(df)

        num_cols = df.select_dtypes(include=['float', 'int']).columns
        if len(num_cols) >= 2:
            x_col = st.selectbox("X", num_cols, index=0)
            y_col = st.selectbox("Y", num_cols, index=1)
            st.plotly_chart(px.scatter(df, x=x_col, y=y_col, hover_data=["run_id"]), use_container_width=True)

    with tab2:
        st.subheader("Correlation between hyperparameters and metrics")
        if df.select_dtypes(include='number').shape[1] >= 2:
            corr = df.select_dtypes(include='number').corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            target = st.selectbox("Target metric", corr.columns)
            top_corr = corr[target].drop(target).sort_values(key=abs, ascending=False).head(3)
            for param in top_corr.index:
                fig = px.scatter(df, x=param, y=target, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("`val_loss` curves per run")
        selected_runs = st.multiselect("Select runs to display loss curves", df["run_id"].tolist(), default=[])
        # df["run_id"].tolist()

        all_curves = []
        for run_id in selected_runs:
            curve = get_loss_curve(run_id)
            if curve is not None:
                all_curves.append(curve)

        if all_curves:
            full_df = pd.concat(all_curves)
            fig = px.line(full_df, x="step", y="value", color="run_id", title="val_loss by step")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loss curves found for the selected runs.")

    with tab4:
        st.subheader("AUC values")

        
        experiments = client.search_experiments()
        experiments_hla = { e.name: e.experiment_id for e in experiments if "no" not in e.name }
        experiments_nohla = { e.name: e.experiment_id for e in experiments if "no" in e.name } 
        exp_name_to_id = { e.name: e.experiment_id for e in experiments }
        
        runs_hla, _ = load_runs([exp_id for exp_name, exp_id in experiments_hla.items()], val_loss_threshold=11.95)
        runs_nohla, _ = load_runs([exp_id for exp_name, exp_id in experiments_nohla.items()], val_loss_threshold=11.95)
        
        selected_run_hla = st.selectbox("Select run w/HLA to display loss curves", runs_hla["run_id"].tolist())
        selected_run_nohla = st.selectbox("Select run wo/HLA to display loss curves", runs_nohla["run_id"].tolist())
        
        run_hla  = client.get_run(selected_run_hla)
        run_nohla  = client.get_run(selected_run_nohla)

        def fix_artifact_uri(artifact_uri):
            artifact_dir = artifact_uri.replace("/homes", "/home")
            artifact_dir = artifact_dir.replace('/nfs/research/birney/users', "/home")
            return artifact_dir


        def get_auc_dfs(run):
            artifact_dir = fix_artifact_uri(run.info.artifact_uri)
            AUCDIR = f"{artifact_dir}/auc/"
            unpooled_auc = pd.read_parquet(f"{AUCDIR}/df_auc_unpooled.parquet").query("n_diseased > 100")
            both_auc = pd.read_parquet(f"{AUCDIR}/df_both.parquet")
            unpooled_auc = unpooled_auc.drop(["auc"], axis=1)
            unpooled_auc = unpooled_auc[~unpooled_auc.duplicated()]
            unpooled_auc = unpooled_auc.drop(['ICD-10 Chapter (short)', 'color', 'auc_variance_delong', 'count'], axis=1)
            return unpooled_auc, both_auc
        
        unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla)
        unpooled_auc_nohla, both_auc_nohla = get_auc_dfs(run_nohla)

        print(run_hla)
        st.text(f"HLA loss: {run_hla.data.metrics['val_loss']:.4f}")
        st.text(f"NO HLA loss: {run_nohla.data.metrics['val_loss']:.4f}")
        kk = pd.merge(unpooled_auc_hla, unpooled_auc_nohla, on=['token', 'age', 'n_healthy', 'n_diseased', 'index', 'name', 'sex'], suffixes=['_hla', '_nohla'])
        kk = kk.query("auc_delong_hla > auc_delong_nohla").assign(diff=lambda x: x.auc_delong_hla - x.auc_delong_nohla).sort_values("diff", ascending=False)
        st.dataframe(kk)
        # st.dataframe(unpooled_auc_hla)
        # st.dataframe(unpooled_auc_nohla)
