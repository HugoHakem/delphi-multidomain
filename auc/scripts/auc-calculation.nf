#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.outdir = "/hps/nobackup/birney/users/bonazzola/auc/work"
output_dir = "/hps/nobackup/birney/users/bonazzola/auc/"
params.n_subject_chunks = 10
params.n_disease_chunks = 10

// Process 1
process computeLogits {    

    tag "${runid}__${subject_chunk}_of_${n_chunks}"

    executor 'slurm'
    cpus 1
    memory '32 GB'
    time '0.2h'    
    array 100

    // errorStrategy 'ignore'

    input:
        path logits_py
        tuple val(runid), val(subject_chunk), val(n_chunks)

    output:
        tuple val(runid), val(subject_chunk), val(n_chunks)

    script:
    """    
    # python ${logits_py} \
    #  --chunk_index ${subject_chunk} \
    #  --n_chunks ${n_chunks} \
    #  --runid ${runid} \
    #  --logits_file ${output_dir}/logits/${runid}/logits_${subject_chunk}_of_${n_chunks}.pt \
    #  --tokens_file ${output_dir}/tokens/${runid}/tokens_${subject_chunk}_of_${n_chunks}.parquet
    
    #test -s ${output_dir}/logits/${runid}/logits_${subject_chunk}_of_${n_chunks}.pt
    #test -s ${output_dir}/tokens/${runid}/tokens_${subject_chunk}_of_${n_chunks}.parquet
    """    
}

// Process 2
process computeIndicesPerDiseaseAgeSex {

    tag "${runid}__${subject_chunk}_of_${n_chunks}__${disease_chunk}_of_${n_disease_chunks}"

    executor 'slurm'
    cpus 1
    memory '16 GB'
    time '0.5h'
    submitRateLimit = "200/1sec"
    array 50

    errorStrategy = { ( task.exitStatus == 0 ) ? "retry" : "terminate" }

    input:
        path indices_py
        tuple val(runid), val(subject_chunk), val(n_chunks), val(disease_chunk), val(n_disease_chunks)

    output:
        tuple val(runid), val(subject_chunk), val(disease_chunk), val(n_disease_chunks)

    script:
    """
    python ${indices_py} \
       --runid ${runid} \
       --tokens_file ${output_dir}/tokens/${runid}/tokens_${subject_chunk}_of_${n_chunks}.parquet \
       --output_file ${output_dir}/indices/${runid}/indices__${subject_chunk}_of_${n_chunks}__${disease_chunk}_of_${n_disease_chunks}.parquet \
       --chunk_index ${subject_chunk} \
       --dchunk ${disease_chunk} \
       --n_dchunks ${n_disease_chunks}

    test -s ${output_dir}/indices/${runid}/indices__${subject_chunk}_of_${n_chunks}__${disease_chunk}_of_${n_disease_chunks}.parquet
    """    
}

// Process 3
process extractLogitsComputeAUC {

    tag "${runid}__${disease_chunk}_of_${n_disease_chunks}"

    executor 'slurm'
    cpus 4
    memory '128 GB'
    time '0.5h'

    errorStrategy = { ( task.exitStatus == 0 ) ? "retry" : "terminate" }

    input:
        path auc_py
        tuple val(runid), val(disease_chunk), val(n_disease_chunks)

    output:
        val(runid)

    script:
    """
    # mkdir -p ${params.outdir}/auc/${runid}
    python $auc_py \
      --runid ${runid} \
      --indices_root ${output_dir}/indices/${runid} \
      --logits_root ${output_dir}/logits/${runid} \
      --logits_merged_outdir ${output_dir}/logits_merged/${runid} \
      --indices_file_pattern "indices__*_of_${params.n_subject_chunks}__${disease_chunk}_of_${n_disease_chunks}.parquet" \
      --logits_file_pattern "logits_*_of_${params.n_subject_chunks}.pt" \
      --auc_output_file ${output_dir}/aucs/${runid}/aucs__${disease_chunk}_of_${n_disease_chunks}.csv \
      --bootstrap \
      --n_bootstrap 200 

    test -s ${output_dir}/aucs/${runid}/aucs__${disease_chunk}_of_${n_disease_chunks}.csv
    """
}

// Process 4
process concatenateAUCs {

    tag "${runid}"

    executor 'slurm'
    cpus 1
    memory '8 GB'
    time '0.1h'

    input:
        val(runid)

    output:
        val(runid)

    script:
    
    """
    set -euo pipefail
    shopt -s nullglob
    files=\$(printf "%s\n" ${output_dir}/aucs/${runid}/aucs__*_of_${params.n_disease_chunks}.csv | sort -V)
    
    if [ -z "\$files" ]; then
        echo "No AUC chunk files found" >&2
        exit 1
    fi

    # concatenate AUC files while keeping only one header
    {
        head -n 1 \$(echo "\$files" | head -n 1)
        echo "\$files" | while read -r f; do tail -n +2 "\$f"; done
    } > aucs.csv
    mv aucs.csv ${output_dir}/aucs/${runid}/aucs.csv

    python - << 'EOF'
import mlflow
import os
        
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        
run_id = "${runid}"
artifact_file = "${output_dir}/aucs/${runid}/aucs.csv"
artifact_subdir = "aucs"
        
assert os.path.exists(artifact_file), f"Missing artifact: {artifact_file}"
        
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(artifact_file, artifact_path=artifact_subdir)
        
print(f"Logged {artifact_file} to run {run_id}")
EOF
    """
}    


workflow {

    // 0) channels
    Channel.fromPath('runs.csv')
           .splitCsv(header: true)
           .map { row -> row.runid }
           .set { runid_ch }

    Channel.from(1..<params.n_subject_chunks+1)
           .set { subject_chunk_ch }

    Channel.from(1..<params.n_disease_chunks+1)
           .set { disease_chunk_ch }

    // ###########################################################################

    // 1) logits
    logits_py = file("s01_compute_logits_per_chunk.py")

    n_subject_chunks = Channel.value(params.n_subject_chunks)
    runid_subject_ch = runid_ch.combine(subject_chunk_ch).combine(n_subject_chunks)
    logits_ch = computeLogits( logits_py, runid_subject_ch )

    // 2) subject × disease
    indices_py = file("s02_compute_case_ctrl_indices.py")
    n_disease_chunks = Channel.value(params.n_disease_chunks)
    indices_input_ch = logits_ch.combine(disease_chunk_ch).combine(n_disease_chunks)
    
    // indices_input_ch.view( idx -> "Computing indices for: $idx")

    indices_ch = computeIndicesPerDiseaseAgeSex( indices_py, indices_input_ch )

    // 3) logit extraction
    auc_py = file("s03_extract_logits_compute_auc.py")
    per_run_disease_ch = indices_ch.groupTuple(by: 0).map{ runid, subject_chunk, disease_chunk, n_disease_chunks -> runid }
    per_run_disease_ch = indices_ch.groupTuple(by: [0, 2]).map{ runid, subject_chunk, disease_chunk, n_disease_chunks -> [runid, disease_chunk, n_disease_chunks[0]] }
    // per_run_ch.view( runid -> "Processing AUC for run ID: $runid")
    
    auc_ch = extractLogitsComputeAUC( auc_py, per_run_disease_ch )

    run_ch = auc_ch.distinct()
    concatenateAUCs( run_ch )

}
