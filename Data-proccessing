#creat.network_matrix
install.packages("NetPreProc")
library(NetPreProc)
netname <- "network.matrix.rda"
load(netname)
ind <- which(rowSums(network != 0) == 1)

min_weight <- min(rowSums(network)[ind]) # min weight among those ind
W <- network # connections
W[network < min_weight] <- 0 # we want to remove noisy connections
cat("connection_range: ", range(rowSums(W != 0)), "\n")
cat("connection_removed ", sum(network != 0) - sum(W != 0), "\n")

cat("W_issymmetric: ", isSymmetric(W), "\n")
W <- Laplacian.norm(W)
cat("W_issymmetric_afternormalization: ", isSymmetric(W), "\n")
#---------------------------------------------------------------------
#creat.lable_annotation_matrix
aspect <- c("C", "P", "F")
org <- c("Homo Sapiens", "Yeast", "Mouse", "D. Melanogaster")
load("label_matrix") # A
ann.num <- colSums(A) # how many annotations exist
to_retain_c <- which(ann.num >= 3 & ann.num <= 200)
cat("number of terms:", ncol(A), "\n")
cat("number of filtered terms: ", length(to_retain_c), "\n")
A <- A[, to_retain_c]
