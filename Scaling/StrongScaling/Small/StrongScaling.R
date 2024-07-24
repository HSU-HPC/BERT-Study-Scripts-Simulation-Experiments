library(BERT)
library(HarmonizR)

options(digits.secs = 6)

# some dummy data so that everything is loaded
dataset <- generate_dataset(1000,10,10,0.1, 2)
dummy <- BERT(dataset, cores=2)

#######
features <- 6000
batches <- 64
samplesperbatch <- 10
classes <- 2
mvfravtion <- 0.1
cores = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
adjustment_method <- c("ComBat", "limma")
repetitions <- 5


findbestparams <- function(){
    features <- 6000
    batches <- 64
    samplesperbatch <- 10
    classes <- 2
    mvfravtion <- 0.1
    corered <- c(2,4,8,16)
    stopParB <- c(2,4,8,16)
    backends <- c("default", "file")
    
    repetitions <- 1
    
    #####
    result_df <- data.frame(Repetition=integer(),
                            cores = integer(),
                            runtime = double(),
                            backend=character(),
                            stopParBatches=integer(),
                            corereduction=integer())
    result_path = "results/params_strong_small.csv"
    
    for(rep in 1:repetitions){
        # generate dataset
        dataset <- generate_dataset(features, batches, samplesperbatch, mvfravtion, classes)
        for(back in backends){
            # BERT, sequential run
            total_start <- Sys.time()
            dataset_adjusted <- BERT(dataset, 1, method="ComBat", verify=FALSE, qualitycontrol=FALSE, backend = back)
            total_end <- Sys.time()
            execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
            result_df[nrow(result_df) + 1,] = c(rep, 1, execution_time, back, -1,-1)
            
            # BERT parallel run
            for(stop in stopParB){
                for(red in corered){
                    print(paste(stop, red, back))
                    total_start <- Sys.time()
                    dataset_adjusted <- BERT(dataset, 16, method="ComBat", verify=FALSE, qualitycontrol=FALSE, backend = back, stopParBatches=stop, corereduction=red)
                    total_end <- Sys.time()
                    execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
                    result_df[nrow(result_df) + 1,] = c(rep, 16, execution_time, back, stop,red)
                    gc()
                }
                # write to file
                write.csv(result_df, result_path)
            }
        }
    }
    return(result_df)
}

config_df = findbestparams()
config_df[which.min(config_df$runtime),]



result_df <- data.frame(Repetition=integer(),
                        cores = integer(), algorithm=character(),
                        method = character(), runtime = double(), uniqueremoval=logical())
result_path = "results/Results_strong_small.csv"

for(rep in 1:repetitions){
  # generate dataset
  dataset <- generate_dataset(features, batches, samplesperbatch, mvfravtion, classes)
  for(core in cores){
    print(typeof(core))
    for(method in adjustment_method){
      # BERT
      total_start <- Sys.time()
      dataset_adjusted <- BERT(dataset, core, method=method, verify=FALSE, qualitycontrol=FALSE, stopParBatches = as.integer(config_df[which.min(config_df$runtime),]$stopParBatches), corereduction=as.integer(config_df[which.min(config_df$runtime),]$corereduction), backend=config_df[which.min(config_df$runtime),]$backend)
      total_end <- Sys.time()
      execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
      result_df[nrow(result_df) + 1,] = c(rep, core, "BERT", method, execution_time, FALSE)
      write.csv(result_df, result_path)
      # HarmonizR
      # order by batch
      dataset_ordered <- dataset[order(dataset$Batch), ]
      dataset_numeric <- t(dataset_ordered[,!names(dataset_ordered) %in% c("Batch", "Label")])
      colnames(dataset_numeric) <- paste("Sub", colnames(dataset_numeric), sep = "_")
      dataset_description <- dataset_ordered["Batch"]
      names(dataset_description)[names(dataset_description) == 'Batch'] <- 'batch'
      dataset_description["sample"] <- 1:(dim(dataset_numeric)[2])
      dataset_description["ID"] <- colnames(dataset_numeric)
      dataset_description <- dataset_description[c("ID", "sample", "batch")]
      dataset_description["batch"] <- as.integer(dataset_description[["batch"]])
      
      dataset_numeric <- as.data.frame(dataset_numeric)
      
      total_start <- Sys.time()
      dataset_adjusted <- harmonizR(dataset_numeric, dataset_description, cores = core, algorithm = method) #  
      total_end <- Sys.time()
      execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
      result_df[nrow(result_df) + 1,] = c(rep, core, "HarmonizR", method, execution_time, TRUE)
      # write to file
      write.csv(result_df, result_path)
      #total_start <- Sys.time()
      #dataset_adjusted <- harmonizR(dataset_numeric, dataset_description, cores = core, algorithm = method, ur=FALSE)
      #total_end <- Sys.time()
      #execution_time <- as.numeric(as.POSIXct(total_end,origin = "1970-01-01")) - as.numeric(as.POSIXct(total_start,origin = "1970-01-01"))
      #result_df[nrow(result_df) + 1,] = c(rep, core, "HarmonizR", method, execution_time, FALSE)
    }
    # write to file
    write.csv(result_df, result_path)
  }
}
