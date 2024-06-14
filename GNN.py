from tensorflow import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
#------------------------------------------
# Load network_matrix from a CSV file
network_matrix = pd.read_csv('triplet.weight.n1.csv')

# Drop the 'Unnamed: 0' column
network_matrix = network_matrix .drop('Unnamed: 0', axis=1)

#load lable_annotation_matrix
ann_matrix_C =pd.read_csv('ann.filt.4932.C.csv', sep="," , header =0)
ann_matrix_C.rename(columns={'Unnamed: 0': 'protID'}, inplace=True)
#------------------------------------------
#define edges in tensorlow library with float
edges = network_matrix[["source", "target"]].to_numpy().T
print("Edges shape:", edges.shape)
edge_weights =network_matrix['weight']
edge_weights = edge_weights.astype('float32')
edge_weights =tf.convert_to_tensor(edge_weights.values, dtype=tf.float32)
#-----------------------------------------
#convert converts the 2D array into a 1D array.
unique_nodes = np.unique(edges.flatten())
#This creates a dictionary that maps each unique node to a unique integer index. 
node_to_int = {node: idx for idx, node in enumerate(unique_nodes)}
#This replaces each node in edges with its corresponding integer index from the 
edges = np.vectorize(node_to_int.get)(edges)

#Convert mapped_edges to int32 dtype
edges = edges.astype(np.int32)
#----------------------------------------
#This line extracts the column names from the DataFrame ann_matrix and stores them in a set. Using a set ensures that the feature names are unique.
feature_names = set(ann_matrix.columns)
#This line calculates the number of unique features (i.e., the number of columns in the DataFrame) and stores it in num_features
num_features = len(feature_names)
#This line sorts the DataFrame ann_matrix based on the values in the "protID" column. The sorted DataFrame is stored in sorted_df
sorted_df = ann_matrix_C.sort_values("protID")
#This line selects only the columns with numeric data types from the sorted DataFrame sorted_df. The resulting DataFrame with numeric columns is stored in numeric_cols.
numeric_cols = sorted_df.select_dtypes(include=[np.number])
#This line converts the DataFrame numeric_cols to a NumPy array and ensures that all the elements in the array are of type float. The resulting NumPy array is stored in data_array.
data_array = numeric_cols.to_numpy().astype(float)
#------------------------------------
#This function, MLP, defines a multi-layer perceptron (MLP) model using the Keras Sequential 
#hidden_units: A list where each element represents the number of units (neurons) in a hidden layer.
#dropout_rate: The rate at which inputs are set to zero during dropout, a regularization technique to prevent overfitting.
#name: An optional name for the Sequential model.
#Initializes an empty list to store the layers of the model.
#Iterates over each element in hidden_units to define the layers:
#layers.BatchNormalization(): Adds a batch normalization layer to normalize the inputs of the next layer, which can help with training stability and performance.
#layers.Dropout(dropout_rate): Adds a dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
#layers.Dense(units, activation=tf.nn.relu): Adds a dense (fully connected) layer with a specified number of units and ReLU activation function
#Constructs a Keras Sequential model from the list of layers and assigns an optional name to it.

def MLP(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.relu))
    return keras.Sequential(fnn_layers, name=name)
  #----------------------------------
#This class GraphConvLayer implements a graph convolutional layer in TensorFlow using Keras. Here's a detailed breakdown of each part of the class and what it does
#__init__ method: This is the constructor method, which initializes the layer with the specified parameters.
#hidden_units: List of hidden units for the MLP layers.
#dropout_rate: Dropout rate to use in the MLP layers.
#aggregation_type: Type of aggregation to use ("mean", "sum", or "max").
#update_type: Type of update function to use. Currently, only MLP is implemented.
#normalize: Boolean flag to indicate whether to normalize the output embeddings.
#*args, **kwargs: Additional arguments passed to the base Layer class.
#Layer Components:
#self.prepare_fun: MLP used for preparing messages from node representations.
#self.update_fun: MLP used for updating node embeddings.
#prepare method: Prepares the messages from the node representations.
#node_repesentations: The representations of the nodes.
#weights: Optional edge weights.
#Message Preparation: The node representations are passed through the prepare_fun MLP. If weights are provided, they are applied to the messages.
#aggregate method: Aggregates messages from the neighbors.
#node_indices: Indices of the nodes.
#neighbour_messages: Messages from the neighbors.
#node_repesentations: Original node representations.
#Aggregation: Depending on the aggregation_type, it performs sum, mean, or max aggregation.
#update method: Updates node embeddings with aggregated messages.
#node_repesentations: Original node representations.
#aggregated_messages: Aggregated messages from neighbors.
#Update: Combines original node representations with aggregated messages and passes through the update_fun MLP. Optionally normalizes the embeddings.
#call method: Processes the inputs to produce node embeddings.
#inputs: A tuple containing node representations, edges, and optional edge weights.
#Processing:
#Extracts node indices and neighbor indices from edges.
#Gathers neighbor representations.
#Prepares neighbor messages.
#Aggregates neighbor messages.
#Updates node embeddings with aggregated messages.

class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        update_type="MLP",
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.update_type = update_type
        self.normalize = normalize

        self.prepare_fun = MLP(hidden_units, dropout_rate)

        if update_type == "MLP":
          self.update_fun = MLP(hidden_units, dropout_rate)
        # else .....

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.prepare_fun(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        # Linear: Add node_repesentations and aggregated_messages.
        #   Matrices B and W ignored
        h = node_repesentations + aggregated_messages


        # Apply the processing function (f_v).
        node_embeddings = self.update_fun(h)

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        # Gather slices from node_repesentations according to neighbour_indices
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)
      #----------------------------------------------
#The GNNNodeClassifier class is a TensorFlow Keras model for node classification in a graph using Graph Neural Networks (GNN). Here’s a detailed breakdown of each part of the class and its functionality
#__init__ method: This constructor initializes the GNNNodeClassifier model with the specified parameters.
#graph_info: A tuple containing node features, edges, and edge weights.
#num_classes: Number of classes for classification.
#hidden_units: List of hidden units for the MLP layers.
#aggregation_type: Type of aggregation to use in the graph convolution layer ("mean", "sum", or "max").
#update_type: Type of update function to use in the graph convolution layer (currently only MLP).
#dropout_rate: Dropout rate for the MLP layers.
#normalize: Boolean flag to indicate whether to normalize the output embeddings.
#*args, **kwargs: Additional arguments passed to the base Model class.
#An MLP layer for preprocessing the node features before applying graph convolutions.
#Two graph convolution layers for learning node embeddings from the graph structure and node features.
#call method: Defines the forward pass of the model.
#input_node_indices: Indices of the nodes for which to compute the embeddings and output class probabilities (represents the batch).
#Preprocessing:
#The node features are preprocessed using the preprocess MLP.
#Graph Convolution Layers:
#The preprocessed node representations are passed through two graph convolution layers.
#Skip connections are commented out, so the output of each convolution layer directly overwrites x.
#Postprocessing:
#The node embeddings are postprocessed using the postprocess MLP.
#Output Computation:
#The embeddings of the nodes specified by input_node_indices are fetched.
#The final output probabilities for each node in the batch are computed using the compute_out dense layer.


class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info, # triple:nodes, edges, weights structures
        num_classes,
        hidden_units,# to be used in function MLP
        aggregation_type="mean",
        update_type="MLP",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features,
        #    edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1. tf.math.reduce_sum()
        #    computes the sum of elements across dimensions of a tensor.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = MLP(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            update_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            update_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = MLP(hidden_units, dropout_rate, name="postprocess")
        # Create a compute output layer.
        self.compute_out = layers.Dense(units=num_classes,
                                        activation="softmax",
                                        name="out")

    def call(self, input_node_indices):
        # input_node_indices represent our ```batch```
        # Preprocess the node_features to produce node representations.
        #  that is, nodes features are pre-embedded before graph convolution
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
          # Skip connection.
          #x = x1 + x
        # no skip connection
        x=x1
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
           # Skip connection.
           #x = x2 + x
        # no skip connection
        x = x2
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        out = self.compute_out(node_embeddings)
        return out
      #-------------------------------------------
      #ann_matrix["protID"].unique(): This retrieves all unique protein IDs from the protID column of the DataFrame ann_matrix.
#sorted(...): Sorts these unique protein IDs.
#enumerate(...): Enumerates the sorted protein IDs, providing both an index (idx) and the protein ID (name).
#np.ones((len(prot_idx), emb_size)): Creates a NumPy array of shape (number_of_proteins, embedding_size), filled with ones. Each protein (node) will have an embedding of size 32.
 #prot_idx = {name: idx for idx, name in enumerate(sorted(ann_matrix["protID"].unique()))}
#tf.cast(node_features, dtype=tf.float32): Converts the NumPy array node_features to a TensorFlow tensor with data type float32.

 emb_size = 32
node_features = np.ones((len(prot_idx), emb_size))#prot_idx or protID(first.col of label.matrix)
node_features = tf.cast(node_features, dtype=tf.float32)
#-------------------------------------------------
#This loop iterates over the range starting from 1 to the number of columns in ann_matrix. It skips the first column, which is presumably protID.
#ann_matrix.iloc[:, kk] extracts the kk-th column of the DataFrame 
#Creates a dictionary mapping each unique value in class_values to a unique index using enumerate
#Calculates the number of unique classes in the current column.
#Creates a dictionary mapping each unique protID to a unique index.

for kk in range(1,ann_matrix.shape[1]):
  class_values = ann_matrix.iloc[:,kk]
  class_idx = {name: id for id, name in enumerate(class_values)}
  num_classes = len(class_idx)
  prot_idx = {name: idx for idx, name in enumerate(sorted(ann_matrix["protID"].unique()))}
  #-----------------------------------------------
#initialization:

#fake_X: A dummy array used to create splits based on the target y.
#kf: A StratifiedKFold instance to ensure each fold has the same proportion of classes.
#Other variables (acc, current_pred_vector, fold, permuting_vector, predict_matrix) are initialized to store results.
#Looping Through Class Columns:

#The loop iterates over the class columns of ann_matrix, starting from the second column.
#Preparing Data for Each Fold:

#For each fold, the training and testing indices are obtained using kf.split.
#The train_idx and test_idx are used to create the training and testing datasets (x_train, x_test, y_train, y_test).
#Model Definition and Compilation:

#A GNNNodeClassifier model is defined for each fold.
#The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
#Training with Early Stopping:

#The model is trained with early stopping to prevent overfitting.
#The fold number is incremented after each training session.
#Prediction and Accuracy Calculation:

#Predictions are made on the test set.
#Predicted labels are appended to current_pred_vector and predict_matrix.
#Accuracy for the current fold is calculated and stored in acc.
#Final Output:

#After all folds are processed, the final accuracies, mean accuracy, and standard deviation of accuracies are printed.
fake_X = np.ones(ann_matrix.shape[0])
kf = StratifiedKFold(n_splits=3)
scaler = StandardScaler()
acc = []
current_pred_vector = []
fold = 1
permuting_vector = []
predict_matrix =[]
for kk in range(1,ann_matrix.shape[1]):
  y= ann_matrix_C.iloc[:, kk].to_numpy()
  for train_idx, test_idx in kf.split(X = fake_X, y = y):
    print(f"fold: {fold}")
    print(f"type(train_idx): {type(train_idx)}")
    x_train, x_test = train_idx, test_idx
    y_train, y_test = y[train_idx], y[test_idx]
    graph_info = (node_features, edges, edge_weights)
    gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
    )
    learning_rate = 0.005
    gnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics="acc"
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss"
    , patience=3, restore_best_weights=True
    )
    num_epochs = 3
    batch_size = 64
    history = gnn_model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        verbose=2,
        batch_size=batch_size,
        callbacks=[early_stopping],
    )
    fold += 1
  permuting_vector.append(x_test)
  y_test_pred = gnn_model.predict(x_test)
  y_test_pred = np.argmax(y_test_pred ,axis=1)
  current_pred_vector.append(y_test_pred)
  predict_matrix.append(current_pred_vector)
  fold_accuracy = np.mean(y_test_pred == y_test)

  acc.append(fold_accuracy)
  print(f"\tmean accuracy:{np.mean(acc)}, sdt accuracy:{np.std(acc)}")








  


