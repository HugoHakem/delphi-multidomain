#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.outdir = "/hps/nobackup/birney/users/bonazzola/auc/work"

process computeLogits {

    tag "${runid}__${subject_chunk}_of_${n_chunks}"

    executor 'slurm'
    cpus 1
    memory '16 GB'
    time '0.1h'

    input:
        path logits_py
        tuple val(runid), val(subject_chunk)
        val(n_chunks)

    output:
        tuple val(runid), val(subject_chunk), path("logits/${runid}/logits_${subject_chunk}_of_${n_chunks}.pt"), path("tokens/${runid}/tokens_${subject_chunk}_of_${n_chunks}.parquet") 

    script:
    """
    mkdir -p ${params.outdir}/logits/${runid}
    mkdir -p ${params.outdir}/tokens/${runid}

    python ${logits_py} \
        --chunk_index ${subject_chunk} \
        --n_chunks ${n_chunks} \
        --runid ${runid} \
        --logits_file logits/${runid}/logits_${subject_chunk}_of_${n_chunks}.pt \
        --tokens_file tokens/${runid}/tokens_${subject_chunk}_of_${n_chunks}.parquet
    """    
}

process computeIndicesPerDiseaseAgeSex {

    tag "${runid}__${subject_chunk}__${disease_chunk}_of_${n_disease_chunks}"

    executor 'slurm'
    cpus 1
    memory '8 GB'
    time '0.1h'

    input:
        path indices_py
        tuple val(runid), val(subject_chunk), path(logits), path(tokens), val(disease_chunk)
        val(n_disease_chunks)

    output:
        tuple val(runid), val(disease_chunk), path("indices/${runid}/indices_${subject_chunk}_${disease_chunk}.parquet")

    script:
    """
    mkdir -p indices/${runid}
    python ${indices_py} \
        --runid ${runid} \
        --tokens_file ${tokens} \
        --output "indices/${runid}/indices_${subject_chunk}_${disease_chunk}.parquet" \
        --chunk_index ${subject_chunk} \
        --dchunk ${disease_chunk} \
        --n_dchunks ${n_disease_chunks}
    """    
}

process extractLogits {

    publishDir 'results/extracted', mode: 'copy'

    input:
        tuple val(runid), val(disease_chunk), path(logits), path(indices)
    output:
        tuple val(runid), val(disease_chunk), path("extracted_${runid}_${subject_chunk}_${disease_chunk}.txt")

    script:
    """
    echo "extract run=${runid} subject=${subject_chunk} disease=${disease_chunk}" \
        > extracted_${runid}_${subject_chunk}_${disease_chunk}.txt
    """
}

process computeAUCs {

    publishDir 'results/auc', mode: 'copy'

    input:
        tuple val(runid), val(disease_chunk), path(extracted_files)

    output:
        path "auc_${runid}_${disease_chunk}.txt"

    script:
    """
    echo "AUC run=${runid} disease=${disease_chunk}" \
        > auc_${runid}_${disease_chunk}.txt
    """
}

params.n_subject_chunks = 100
params.n_disease_chunks = 10

workflow {

    Channel.fromPath('runs.csv')
           .splitCsv(header: true)
           .map { row -> row.runid }
           //.view { kk -> "Run ID: $kk"}
           .set { runid_ch }

    Channel.from(1..<params.n_subject_chunks+1)
           //.view(subject_chunk_index -> "Subject index: $subject_chunk_index")    
           .set { subject_chunk_ch }

    Channel.from(1..<params.n_disease_chunks+1)
           //.view(disease_chunk_index -> "Disease index: $disease_chunk_index")    
           .set { disease_chunk_ch }

    
    runid_subject_ch = runid_ch.combine(subject_chunk_ch)

    // 1) logits
    logits_py = file("s01_compute_logits_per_chunk.py")
    logits_ch = computeLogits(logits_py, runid_subject_ch, params.n_subject_chunks)

    // 2) subject × disease
    indices_py = file("s02_compute_case_ctrl_indices.py")
    indices_input_ch = logits_ch.combine(disease_chunk_ch)
    indices_ch = computeIndicesPerDiseaseAgeSex(indices_py, indices_input_ch, params.n_disease_chunks)

    // 3) logit extraction
    // runid, disease_chunk_ch, extracted_logits_ch = extractLogits(runid_ch, disease_chunk_ch, logits_ch, indices_ch)

    // 4) AUC: sync por (runid, disease)
    // computeAUCs(runid, disease_chunk_ch, extracted_logits_ch)

}
