# vergleich limma, ComBat, with and without covariables
library(BERT)

strip_Covariable <- function(dataset){
    ds <- dataset[, !names(dataset) %in% c("Cov_1")] 
    return(ds)
}

#######
features <- 600
batches <- 8
samplesperbatch <- 100
imbalance <- 0.3
mv <- 0.1
core <- 1
adjustment_method <- c("ComBat", "limma")
repetitions <- 10
refs = c(2,4,6)

result_path = "results/Results_cov_ref.csv"
#####
result_df <- data.frame(Repetition=integer(), method = character(), numref=integer(), ASWLabel = double(), ASWBatch = double(), cov=logical())

is_adjustable <- function(data){
  batches = unique(dataset$Batch)
  cov_levels = unique(data$Cov_1)
  for(b in batches){
    data_batch <- data[data["Batch"]==b,]
    for(c in cov_levels){
      data_c <- data_batch[data_batch["Cov_1"]==c, ]
      if(dim(data_c)[1]<(max(refs)/2+2)){ # at least two non-reference sample per class
        return(FALSE)
      }
    }
  }
  return(TRUE)
}

for(rep in 1:repetitions){
  suitable_ds <- FALSE
  while(!suitable_ds){
    dataset <- generate_data_covariables(features, batches, samplesperbatch, mv, imbalance)
    suitable_ds <- is_adjustable(dataset)
  }
  
  # without covariable
  dataset_nocov <- strip_Covariable(dataset)
  # append raw information
  asws <- compute_asw(dataset)
  result_df[nrow(result_df) + 1,] = c(rep, "raw", 0, asws$Label, asws$Batch, FALSE)
  
  # naive adjustment
  for(method in adjustment_method){
    # no covariable
    dataset_adjusted <- BERT(dataset_nocov, core, method=method, qualitycontrol=FALSE)
    asws <- compute_asw(dataset_adjusted)
    result_df[nrow(result_df) + 1,] = c(rep, method, 0, asws$Label, asws$Batch, FALSE)
    # with covariable
    dataset_adjusted <- BERT(dataset, core, method=method, qualitycontrol=FALSE)
    asws <- compute_asw(dataset_adjusted)
    result_df[nrow(result_df) + 1,] = c(rep, method, 0, asws$Label, asws$Batch, TRUE)
  }
  
  # reference adjustment
  for(noref in refs){
    dataset_nocov["Reference"] = 0
    unique_batches = unique(dataset_nocov$Batch)
    # randomly select references per batch and class
    for(b in unique_batches){
      ref_c1 = sample(which((dataset_nocov$Batch==b)&(dataset_nocov$Label==1)), noref/2)
      ref_c2 = sample(which((dataset_nocov$Batch==b)&(dataset_nocov$Label==2)), noref/2)
      dataset_nocov[ref_c1, "Reference"] = 1
      dataset_nocov[ref_c2, "Reference"] = 2
    }
    dataset_adjusted <- BERT(dataset_nocov, core, method="ref", qualitycontrol=FALSE)
    asws <- compute_asw(dataset_adjusted)
    result_df[nrow(result_df) + 1,] = c(rep, "References", noref, asws$Label, asws$Batch, FALSE)
  }
  write.csv(result_df, result_path)
}