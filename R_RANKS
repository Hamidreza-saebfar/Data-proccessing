install.packages("RANKS")
library(RANKS)
input<-tempdir()
results<-tempdir()
save(network_matrix, file=paste(input,"network_matrix.rda",sep=""));
save(label_matrix, file=paste(input,"label_matrix.rda",sep=""));
outputRANKS <- do.RANKS(score = eav.score,kernel = rw.kernel,a = 2,p = 1,seed = 0, rep = 3,
kk = 3,sparsify = FALSE,data.dir = input,
labels.dir = input,data = "/network",output.dir = results,
  labels = "/labels")
