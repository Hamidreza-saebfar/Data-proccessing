library(COSNet)
aspect <- c("C", "P", "F")
org <- c("Homo Sapiens", "Yeast", "Mouse", "D. Melanogaster")
load(paste0("ann.filt.", org ,".",aspect ,".rda" ))#A
load(paste0("string_network.", org ,".n1.rda") )#W
res <- cosnet.cross.validation(B, W_yst, nfolds = 3, cost = 10^-4)
save(res , file = paste0("cosnet_predictions.", org ,".",aspect ,".rda"))
#load(file = paste0("cosnet_predictions.", org ,".",aspect ,".rda"))
