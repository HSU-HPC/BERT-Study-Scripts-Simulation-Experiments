library(BERT)
library(HarmonizR)

options(digits.secs = 6)

# some dummy data so that everything is loaded
dataset <- generate_dataset(1000,10,10,0.1, 2)
dummy <- BERT(dataset, cores=2)

#######
features <- 6000
batches <- c(seq(2,40,by=2), 52, 64)
samplesperbatch <- 10
classes <- 2
mv <- 0.2
cores = c(1) # , 2, 4, 8, 16
adjustment_method <- c("ComBat", "limma")
repetitions <- 10

result_path = "results/batch_scaling.csv"
#####
result_df <- data.frame(Repetition=integer(), batches = integer(),
                        cores = integer(), algorithm=character(),
                        method = character(), runtime = double(),
                        ASWLabel = double(), ASWBatch = double(),
                        numeric = integer())
#####

for (rep in 1:repetitions) {
  for (b in batches) {
    # generate dataset
    dataset <- generate_dataset(features, b, samplesperbatch, mv, classes)
    # append raw information
    asws <- compute_asw(dataset)
    numericValues <- count_existing(dataset)
    result_df[nrow(result_df) + 1,] = c(rep, b, 0, "raw", "raw", 0.0, asws$Label, asws$Batch, numericValues)
    for (core in cores) {
      for (method in adjustment_method) {
        # BERT
        total_start <- Sys.time()
        dataset_adjusted <- BERT(dataset, core, method=method, qualitycontrol = FALSE)
        total_end <- Sys.time()
        asws <- compute_asw(dataset_adjusted)
        numericValues <- count_existing(dataset_adjusted)
        execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
        result_df[nrow(result_df) + 1,] = c(rep, b, core, "BERT", method, execution_time, asws$Label, asws$Batch, numericValues)
        
        # HarmonizR no blocking
        # order by batch
        dataset_ordered <- dataset[order(dataset$Batch), ]
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
        result_df[nrow(result_df) + 1,] = c(rep, b, core, "HarmonizR", method, execution_time, asws$Label, asws$Batch, numericValues)
      }
    }
    # write to file
    write.csv(result_df, result_path)
  }
}