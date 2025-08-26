library(dplyr)
library(tidyr)
library(arrow)
library(biomaRt)
library(stringr)

load_bim <- function(bim_file="./ukb22828_allChr_b0_v3_maf01_04_merge.bim") {
  read.delim(bim_file, sep ='\t', header = FALSE, col.names = c("chrom", "snp", "cm", "pos", "allele1", "allele2"))
}

get_gene_info <- function() {
  
  if (file.exists("gene_info_hg37.csv")) {
    gene_info <- read.csv("gene_info_hg37.csv")
  } else {
    mart <- useMart("ENSEMBL_MART_ENSEMBL",
                    dataset = "hsapiens_gene_ensembl",
                    host = "https://grch37.ensembl.org")
    
    attrs <- c("external_gene_name", "chromosome_name", "transcription_start_site", "transcription_start", "transcription_end", "strand")
    attrs <- c("external_gene_name", "chromosome_name", "start_position", "end_position", "strand")
    
    gene_info <- getBM(attributes = attrs, mart = mart)
    write.csv(gene_info, "gene_info_hg37.csv", row.names = FALSE)
  }
  
  gene_info
  
}


get_snps_near_gene <- function(gene_name, window = 50000) {
  
  gene_info <- get_gene_info() %>% filter(external_gene_name == gene_name)
  
  if (nrow(gene_info) == 0) return(bim_df[0, ])  # returns empty df with columns
  
  row <- gene_info[1, ]
  chr <- as.character(row$chromosome_name)
  strand <- row$strand
  
  tss <- if (strand == 1) row$start_position else row$end_position
  tes <- if (strand == 1) row$end_position else row$start_position
  
  region_start <- min(tss, tes) - window
  region_end   <- max(tss, tes) + window
  
  bim_df %>%
    filter(chrom == chr, pos >= region_start, pos <= region_end) %>%
    mutate(
      dist_genic = case_when(
        pos >= min(tss, tes) & pos <= max(tss, tes) ~ 0,
        strand == 1 & pos < tss ~ -(tss - pos),   # downstream = negative
        strand == 1 & pos > tes ~  (pos - tes),   # upstream = positive
        strand == -1 & pos > tss ~ -(pos - tss),  # downstream = negative (on reverse strand)
        strand == -1 & pos < tes ~  (tes - pos),  # upstream = positive
        TRUE ~ NA_real_
      ),
      tss = tss,
      tes = tes
    )
}

get_significant_snps <- function(
    ages = NULL,
    return_only_snp_names = TRUE,
    pval_thresh = 5e-8,
    path = "minP_per_age.csv"
) {
  # Read CSV (header = TRUE assumes first row has column names)
  df <- read.csv(path, stringsAsFactors = FALSE)
  colnames(df) <- c("SNP", "age20", "age30", "age40", "age50", "age60", "overall")
  
  if (is.null(ages)) {
    # Default: filter by "overall"
    signif <- df[df$overall < pval_thresh, c("SNP", "overall")]
    signif$age <- "overall"
  } else {
    # Build the column names for the requested ages
    age_cols <- paste0("age", ages)
    
    # Subset just SNP + those age cols
    subdf <- df[c("SNP", age_cols)]
    
    # Reshape to long format manually
    signif <- data.frame()
    for (col in age_cols) {
      tmp <- data.frame(
        SNP  = subdf$SNP,
        pval = subdf[[col]],
        age  = col,
        stringsAsFactors = FALSE
      )
      signif <- rbind(signif, tmp[tmp$pval < pval_thresh, ])
    }
  }
  
  if (return_only_snp_names) {
    return(unique(signif$SNP))
  } else {
    return(signif)
  }
}


query_gwas <- function(
    ages      = seq(20, 60, 10),
    base_path = "parquet_outputs/gwas_summary_age%d_all_groups_wide.parquet",
    snps      = NULL,
    metrics   = "P", # you can also query "BETA" or both, c("P", "BETA")
    dims      = NULL,
    sexes     = NULL,
    base_cols = c("SNP","CHR","BP","A1"),
    chunk_size = 100
) {
  if (is.null(ages)) {
    files <- Sys.glob(sprintf(base_path, "*"))
    ages  <- as.integer(gsub("\\D", "", basename(files)))
  }
  
  metrics <- toupper(metrics)
  if (!is.null(sexes)) sexes <- tolower(sexes)
  
  dfs <- lapply(ages, function(age) {
    
    path <- sprintf(base_path, age)
    if (!file.exists(path)) stop("File not found: ", path)
    
    ds <- arrow::open_dataset(path, format = "parquet")
    nms <- names(ds)
    
    metrics_pat <- paste(metrics, collapse = "|")
    dims_pat    <- if (is.null(dims))  "\\d+" else paste(unique(dims), collapse = "|")
    sexes_pat   <- if (is.null(sexes)) "(all|females|males)" else paste(unique(sexes), collapse = "|")
    
    select_regex <- sprintf("^(%s)_(%s)_white_(%s)$", metrics_pat, dims_pat, sexes_pat)
    metric_cols  <- grep(select_regex, nms, value = TRUE)
    
    cols_to_keep <- unique(c(intersect(base_cols, nms), metric_cols))
    
    if (is.null(snps)) {
      df <- read_parquet(path, col_select = any_of(cols_to_keep))
    } else {
      # --- chunk the SNPs
      snp_chunks <- split(snps, ceiling(seq_along(snps) / chunk_size))
      dfs_chunks <- lapply(snp_chunks, function(chunk) {
        ds %>%
          dplyr::filter(SNP %in% chunk) %>%
          dplyr::select(dplyr::any_of(cols_to_keep)) %>%
          collect()
      })
      df <- bind_rows(dfs_chunks)
    }
    
    df_long <- df %>%
      pivot_longer(
        cols = all_of(metric_cols),
        names_to = c("metric","dim","race","sex"),
        names_pattern = "^(BETA|P)_(\\d+)_([A-Za-z]+)_(all|females|males)$",
        values_to = "value"
      ) %>%
      mutate(dim    = as.integer(dim),
             sex    = tolower(sex),
             metric = toupper(metric),
             age    = age) %>%
      dplyr::select(all_of(base_cols), metric, dim, sex, age, value)
    
    df_long
  })
  
  bind_rows(dfs)
}



query_gwas2 <- function(
    ages      = seq(20, 60, 10),
    base_path = "parquet_outputs/gwas_summary_age%d_all_groups_wide.parquet",
    snps      = NULL,
    metrics   = c("P"),
    dims      = NULL,
    sexes     = NULL,
    base_cols = c("SNP","CHR","BP","A1")
) {
  if (is.null(ages)) {
    files <- Sys.glob(sprintf(base_path, "*"))
    ages  <- as.integer(gsub("\\D", "", basename(files)))
  }
  
  metrics <- toupper(metrics)
  if (!is.null(sexes)) sexes <- tolower(sexes)
  
  dfs <- lapply(ages, function(age) {
    
    path <- sprintf(base_path, age)
    if (!file.exists(path)) stop("File not found: ", path)
    
    ds <- arrow::open_dataset(path, format = "parquet")
    # sch <- arrow::parquet_file(path)$schema
    nms <- names(ds)
    
    metrics_pat <- paste(metrics, collapse = "|")
    dims_pat    <- if (is.null(dims))  "\\d+" else paste(unique(dims), collapse = "|")
    sexes_pat   <- if (is.null(sexes)) "(all|females|males)" else paste(unique(sexes), collapse = "|")
    
    select_regex <- sprintf("^(%s)_(%s)_white_(%s)$", metrics_pat, dims_pat, sexes_pat)
    metric_cols  <- grep(select_regex, nms, value = TRUE)
    
    cols_to_keep <- unique(c(intersect(base_cols, nms), metric_cols))
    
    if (is.null(snps)) {
      # Only read necessary columns
      df <- read_parquet(path, col_select = any_of(cols_to_keep))
    } else {
      # if you want to filter by rows (SNPs): use open_dataset + filter
      ds <- open_dataset(path, format = "parquet")
      df <- ds %>%
        dplyr::filter(SNP %in% snps) %>%
        dplyr::select(dplyr::any_of(cols_to_keep)) %>%
        collect()
    }
    
    df_long <- df %>%
      pivot_longer(
        cols = all_of(metric_cols),
        names_to = c("metric","dim","race","sex"),
        names_pattern = "^(BETA|P)_(\\d+)_([A-Za-z]+)_(all|females|males)$",
        values_to = "value"
      ) %>%
      mutate(dim    = as.integer(dim),
             sex    = tolower(sex),
             metric = toupper(metric),
             age    = age) %>%
      dplyr::select(dplyr::all_of(base_cols), metric, dim, sex, age, value)
    
    df_long
  })
  
  bind_rows(dfs)
}


bim_df <- load_bim()