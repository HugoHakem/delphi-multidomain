suppressPackageStartupMessages({
  library(dplyr)
  library(arrow)
  library(stringr)
})

setwd(GWAS_DATA_DIR <- "/homes/bonazzola/Delphi/gwas")

load_bim <- function(bim_file="./ukb22828_allChr_b0_v3_maf01_04_merge.bim") {
  read.delim(bim_file, sep ='\t', header = FALSE, col.names = c("chrom", "snp", "cm", "pos", "allele1", "allele2"))
}

bim_df = load_bim()

get_dataset <- function(age, group = c("all", "males", "females")) {
  group <- match.arg(group)
  file_path <- glue::glue("parquets_{age}/gwas_summary_white_{group}_optimized.parquet")
  arrow::open_dataset(file_path)
}


get_gene_info <- function() {

  library(biomaRt)
  mart <- useMart("ENSEMBL_MART_ENSEMBL",
                  dataset = "hsapiens_gene_ensembl",
                  host = "https://grch37.ensembl.org")
  
  attrs <- c("external_gene_name", "chromosome_name", "start_position", "end_position", "strand")
  
  gene_info <- getBM(attributes = attrs, mart = mart)
  
  gene_info
}


get_snps_near_gene <- function(gene_name, window = 50000) {
  if (file.exists("gene_info_hg37.csv")) {
    gene_info <- read.csv("gene_info_hg37.csv")
  } else {
    gene_info <- get_gene_info()
    write.csv(gene_info, "gene_info_hg37.csv", row.names = FALSE)
  }
  
  gene_info <- gene_info %>% filter(external_gene_name == gene_name)
  if (nrow(gene_info) == 0) return(bim_df[0, ])
  
  row <- gene_info[1, ]
  chr <- as.character(row$chromosome_name)
  strand <- row$strand
  
  tss <- if (strand == 1) row$start_position else row$end_position
  tes <- if (strand == 1) row$end_position   else row$start_position
  
  region_start <- min(tss, tes) - window
  region_end   <- max(tss, tes) + window
  
  bim_df %>%
    filter(chrom == chr, pos >= region_start, pos <= region_end) %>%
    mutate(
      dist_genic = case_when(
        pos >= min(tss, tes) & pos <= max(tss, tes) ~ 0,
        strand == 1  & pos < tss ~ -(tss - pos),   # downstream = negativo
        strand == 1  & pos > tes ~  (pos - tes),   # upstream = positivo
        strand == -1 & pos > tss ~ -(pos - tss),  # downstream = negativo (en reverse strand)
        strand == -1 & pos < tes ~  (tes - pos),  # upstream = positivo
        TRUE ~ NA_real_
      ),
      tss = tss,
      tes = tes
    )
}


query_gwas <- function(
    snps = NULL,
    embedding_dims = NULL,
    ages = c(20, 30, 40, 50, 60),
    groups = c("all", "male", "female"),
    pval_thresh = NULL,
    collect_all = TRUE
) {
  results <- list()
  groups <- match.arg(groups, several.ok = TRUE)
  
  for (group in groups) {
    for (age in ages) {
      ds <- get_dataset(age, group)
      q <- ds
      
      if (!is.null(snps)) q <- q %>% filter(SNP %in% snps)
      if (!is.null(embedding_dims)) q <- q %>% filter(embedding %in% embedding_dims)
      if (!is.null(pval_thresh)) q <- q %>% filter(P < pval_thresh)
      
      q <- q %>% mutate(age = age, group = group)
      
      if (collect_all) {
        results[[paste0(group, "_", age)]] <- q %>% collect()
      } else {
        results[[paste0(group, "_", age)]] <- q  # Lazy query
      }
    }
  }
  
  bind_rows(results)
}
