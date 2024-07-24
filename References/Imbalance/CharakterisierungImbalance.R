# Vergleich mit naivem ComBat/Limma
library(BERT)

nclasses <- 2
nsamplesperclass <- 50
nfeatures <- 600
mv <- 0.1
batches <- 2
repetitions <- 10
norefs = c(2,4,8)
drop = seq(1,35,by=2)

result_df <- data.frame(Repetition=integer(),
                        numa1 = integer(),
                        numa2 = integer(),
                        numb1 = integer(),
                        numb2 = integer(),
                        method = character(),
                        references = integer(),
                        ASWLabel = double(),
                        ASWBatch = double())
result_path = "results/Results_imb.csv"

for(rep in 1:repetitions){
  dataset <- generate_dataset(nfeatures, batches, nsamplesperclass*nclasses, mv, nclasses, deterministic = TRUE)
  dataset["Reference"] = 0
  for(diff in drop){
    # create copy of data
    data = data.frame(dataset)
    # select indices to drop
    batch1_a = which((data$Label==1)&(data$Batch==1))
    batch2_b = which((data$Label==2)&(data$Batch==2))
    dropa = sample(batch1_a, diff, replace=FALSE)
    dropb = sample(batch2_b, diff, replace=FALSE)
    droptotal = c(dropa, dropb)
    # new df
    data = data[-droptotal,]
    
    # naive adjustment
    y_bert_ref <- BERT(data[, -which(names(data) == "Reference")], method="limma", qualitycontrol=FALSE)
    # compute asw
    asws <- compute_asw(y_bert_ref)
    # append to df
    result_df[nrow(result_df) + 1,] = c(rep, 50-diff, 50, 50, 50-diff, "Limma", 0, asws$Label, asws$Batch)
    
    # naive adjustment
    y_bert_ref <- BERT(data[, -which(names(data) == "Reference")], method="ComBat", qualitycontrol=FALSE)
    # compute asw
    asws <- compute_asw(y_bert_ref)
    # append to df
    result_df[nrow(result_df) + 1,] = c(rep, 50-diff, 50, 50, 50-diff, "ComBat", 0, asws$Label, asws$Batch)
    
    for(ref in norefs){
      batch1refa = sample(which((data$Label==1)&(data$Batch==1)), ref/2, replace=FALSE)
      batch1refb = sample(which((data$Label==2)&(data$Batch==1)), ref/2, replace=FALSE)
      batch2refa = sample(which((data$Label==1)&(data$Batch==2)), ref/2, replace=FALSE)
      batch2refb = sample(which((data$Label==2)&(data$Batch==2)), ref/2, replace=FALSE)
      
      data[c(batch1refa, batch2refa), "Reference"] = 1
      data[c(batch1refb, batch2refb), "Reference"] = 2
      
      # perform adjustment
      y_bert_ref <- BERT(data, method="ref", qualitycontrol=FALSE)
      # compute asw
      asws <- compute_asw(y_bert_ref)
      # append to df
      result_df[nrow(result_df) + 1,] = c(rep, 50-diff, 50, 50, 50-diff, "Reference", ref, asws$Label, asws$Batch)
    }
  }
  write.csv(result_df, result_path)
}