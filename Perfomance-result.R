#precrec: This package provides tools for computing and visualizing ROC and PRC.
#PerfMeas: This package includes tools for evaluating the performance of classifiers.
#m: Number of columns (models/predictions) in the scores matrix.
#auc, auprc: Vectors to store the AUC for the ROC and PRC for each column.
#names(auc), names(auprc): Setting the names of the vectors to the column names of the scores matrix for easier identification later.
#Loop through each column of the scores and labels matrices.
#Convert the labels for the current column to a vector and ensure all negative labels are set to 0.
#Use evalmod to evaluate the model, which computes the ROC and PRC.
#Extract the AUC values for both ROC and PRC and store them in the respective vectors.

install.packages('precrec', dependencies=TRUE)
install.packages("PerfMeas")
scores <- score_prediction_label_matrix
labels <- label_matrix

m <- ncol(scores)
auc <- auprc <- rep(0, m)

names(auc) <- names(auprc) <- colnames(scores)

for(k in 1:m) {
  labeling <- as.vector(labels[, k])
  labeling[labeling < 0] <- 0
  tmp_Hs.C <- evalmod(scores = scores[, k], labels = labeling)
  tmp2_Hs.C <- auc(tmp_Hs.C) # auc here is a function
  auc[k] <- tmp2_Hs.C[tmp2_Hs.C$curvetypes == "ROC", "aucs"]
  auprc[k] <- tmp2_Hs.C[tmp2_Hs.C$curvetypes == "PRC", "aucs"]
}
