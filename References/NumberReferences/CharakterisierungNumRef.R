# Charakterisierung bzgl. Zahl an Referenzen/ Referenzklassen

library(BERT)

nclassesmax <- 10 + 2 # 2 classes to co-adjust
nsamplesperclass <- nclassesmax - 2
nfeatures <- 6000
mv <- 0.1
batches <- 2
repetitions <- 10

result_df <- data.frame(Repetition=integer(),
                        classes = integer(),
                        samples = integer(),
                        ASWLabel = double(),
                        ASWBatch = double())
result_path = "results/Results_numref.csv"

for (rep in 1:repetitions) {
  dataset <- generate_dataset(nfeatures, batches, nsamplesperclass*nclassesmax, mv, nclassesmax, deterministic = TRUE)
  for(refcl in 1:(nclassesmax-2)){
    for(refs in 1:nsamplesperclass){
      if((refcl==1) && (refs==1)){
        next
      }
      # create copy of original data
      data = data.frame(dataset)
      data["Reference"] = 0
      # grab the reference classes
      refclasses <- sample(1:(nclassesmax-2), refcl, replace = FALSE)
      # assign the corresponding samples as references
      for(c in refclasses){
        for(b in 1:batches){
          classindices = which((data$Label == c)&(data$Batch == b))
          refindices = sample(classindices, refs, replace=FALSE)
          data[refindices, "Reference"] = c
        }
      }
      
      # perform adjustment
      y_bert_ref <- BERT(data, method="ref", qualitycontrol=FALSE)
      # reduce to the two co-adjusted classes
      y_bert_ref = y_bert_ref[y_bert_ref[["Label"]]%in%c(nclassesmax, nclassesmax-1),]
      asws <- compute_asw(y_bert_ref)
      # append to dataframe
      result_df[nrow(result_df) + 1,] = c(rep, refcl, refs, asws$Label, asws$Batch)
    }
  }
  write.csv(result_df, result_path)
}