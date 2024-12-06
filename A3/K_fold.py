import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
from operations import *
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

# Load data
X, y = load_dataset("p2/data/wine_quality.csv", "quality")

# Set parameters for cross-validation
k = 5
epochs = 500
learning_rate = 0.001

# Define model architecture
n_features = X.shape[1]
layer_sizes = [16, 32, 64, 1]  # Layer sizes (choice)
activations = [ReLU(), ReLU(), Sigmoid(), Identity()]  # Activation functions for hidden and output layers
loss_function = MeanSquaredError()

# Shuffle the data to ensure randomness in the folds
X, y = shuffle(X, y, random_state=42)

# Set up k-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results
fold_errors = []  # MAE for each fold
fold_losses = []  # Training loss for each epoch for each fold

# Perform k-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training on fold {fold + 1}/{k}...")

    # Split data into training and validation sets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Initialize a new neural network for each fold
    net = NeuralNetwork(n_features, layer_sizes, activations, loss_function, learning_rate=learning_rate)
    
    # Train the network on the training data for the specified number of epochs
    _, epoch_losses = net.train(X_train, y_train, epochs)
    fold_losses.append(epoch_losses)

    # Evaluate the model on the validation set
    fold_mae = net.evaluate(X_val, y_val, mean_absolute_error)
    fold_errors.append(fold_mae)

# Compute average and standard deviation of MAE across all folds
mean_mae = np.mean(fold_errors)
std_mae = np.std(fold_errors)
print(f"Average MAE across folds: {mean_mae:.4f}")
print(f"Standard deviation of MAE across folds: {std_mae:.4f}")

# Calculate the average training loss across all folds for each epoch
average_epoch_losses = np.mean(fold_losses, axis=0)

# Plot the average training loss per epoch across all folds
plt.plot(np.arange(epochs), average_epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Average Training Loss across Folds")
plt.show()
