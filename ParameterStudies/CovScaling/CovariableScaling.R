library(BERT)
library(HarmonizR)

strip_Covariable <- function(dataset){
    ds <- dataset[, !names(dataset) %in% c("Cov_1")] 
    return(ds)
}

options(digits.secs = 6)

# some dummy data so that everything is loaded
dataset <- generate_dataset(1000,10,10,0.1, 2)
dummy <- BERT(dataset, cores=2)
#######
features <- 6000
batches <- 8
samplesperbatch <- 80
imbalance <- seq(0.1, 0.5, length.out = 3)
mv <- 0.1
core <- 1
adjustment_method <- c("ComBat", "limma")
repetitions <- 10

result_path = "results/covariables.csv"

#####
result_df <- data.frame(Repetition=integer(), imbalance = double(),
                        algorithm=character(),
                        method = character(), runtime = double(),
                        ASWLabel = double(), ASWBatch = double())
#####


is_adjustable <- function(data){
  batches = unique(dataset$Batch)
  cov_levels = unique(data$Cov_1)
  for(b in batches){
    data_batch <- data[data["Batch"]==b,]
    for(c in cov_levels){
      data_c <- data_batch[data_batch["Cov_1"]==c, ]
      if(dim(data_c)[1]<2){
        return(FALSE)
      }
    }
  }
  return(TRUE)
}

for (rep in 1:repetitions) {
  for (imb in imbalance) {
    
    suitable_ds <- FALSE
    while(!suitable_ds){
      dataset <- generate_data_covariables(features, batches, samplesperbatch, mv, imb)
      suitable_ds <- is_adjustable(dataset)
    }
    
    # without covariable
    dataset_nocov <- strip_Covariable(dataset)
    # append raw information
    asws <- compute_asw(dataset)
    numericValues <- count_existing(dataset)
    result_df[nrow(result_df) + 1,] = c(rep, imb, "raw", "raw", 0.0, asws$Label, asws$Batch)
      for (method in adjustment_method) {
        # BERT with Cov
        total_start <- Sys.time()
        dataset_adjusted <- BERT(dataset, core, method=method, qualitycontrol=FALSE)
        total_end <- Sys.time()
        asws <- compute_asw(dataset_adjusted)
        numericValues <- count_existing(dataset_adjusted)
        execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
        result_df[nrow(result_df) + 1,] = c(rep, imb, "BERT_Cov", method, execution_time, asws$Label, asws$Batch)
        
        # BERT without Cov
        total_start <- Sys.time()
        dataset_adjusted <- BERT(dataset_nocov, core, method=method, qualitycontrol=FALSE)
        total_end <- Sys.time()
        asws <- compute_asw(dataset_adjusted)
        numericValues <- count_existing(dataset_adjusted)
        execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
        result_df[nrow(result_df) + 1,] = c(rep, imb, "BERT", method, execution_time, asws$Label, asws$Batch)
        
        # HarmonizR
        # order by batch
        dataset_ordered <- dataset_nocov[order(dataset_nocov$Batch), ]
        dataset_numeric <- t(dataset_ordered[,!names(dataset_ordered) %in% c("Batch", "Label")])
        dataset_description <- dataset_ordered["Batch"]
        names(dataset_description)[names(dataset_description) == 'Batch'] <- 'batch'
        dataset_description["sample"] <- 1:(dim(dataset_numeric)[2])
        dataset_description["ID"] <- colnames(dataset_numeric)
        dataset_description <- dataset_description[c("ID", "sample", "batch")]
        
        dataset_numeric <- as.data.frame(dataset_numeric)
        
        total_start <- Sys.time()
        dataset_adjusted <- harmonizR(dataset_numeric, dataset_description, cores = core, algorithm = method)
        total_end <- Sys.time()
        dataset_adjusted <- t(dataset_adjusted)
        n <- rownames(dataset_adjusted)
        dataset_adjusted <- data.frame(dataset_adjusted)
        dataset_adjusted["Batch"] <- dataset_description[n, "batch"]
        dataset_adjusted["Label"] <- dataset_ordered[n, "Label"]
        asws <- compute_asw(dataset_adjusted)
        numericValues <- count_existing(dataset_adjusted)
        execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
        result_df[nrow(result_df) + 1,] = c(rep, imb, "HarmonizR", method, execution_time, asws$Label, asws$Batch)
      }
  }
  # write to file
  write.csv(result_df, result_path)
}