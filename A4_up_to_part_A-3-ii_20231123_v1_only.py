# Load the tensors
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast

print("Loading tensors...")
train_reviews_tensor = torch.load('train_reviews_tensor.pt')
val_reviews_tensor = torch.load('val_reviews_tensor.pt')
train_labels_tensor = torch.load('train_labels_tensor.pt')
val_labels_tensor = torch.load('val_labels_tensor.pt')

print( "Recreate the datasets...")
train_dataset = TensorDataset(train_reviews_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_reviews_tensor, val_labels_tensor)

print("Recreate the DataLoaders...")
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Iterate over training DataLoader...")
for i, batch in enumerate(train_loader):
    reviews, labels = batch  # Unpack the tuple directly
    print(f"Batch {i+1}")
    print(f"Review batch shape: {reviews.shape}")
    print(f"Label batch shape: {labels.shape}")
    # Add a break to stop after the first batch for demonstration purposes
    if i == 0: 
        break

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        # Parameters
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN Layer (choose between nn.RNN, nn.LSTM, or nn.GRU)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           dropout=drop_prob, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_size)

        # Activation function (e.g., sigmoid for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        # Embedding and RNN
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        
        # Max pooling and average pooling
        out_max = torch.max(out, dim=1)[0]
        out_avg = torch.mean(out, dim=1)
        out = torch.cat([out_max, out_avg], dim=1)

        # Dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        # Sigmoid function
        sig_out = self.sigmoid(out)

        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        # Check for MPS availability and use it if available
        if torch.backends.mps.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("mps"),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("mps"))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

print("Instantiate the model with specific parameters...")

vocab_size = 1000  # Size of vocabulary obtained from the training data
output_size = 1    # Binary classification (Positive/Negative)
embedding_dim = 400
hidden_dim = 256
n_layers = 2

model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# Optionally, move the model to MPS if available
if torch.backends.mps.is_available():
    model.to("mps")


def log_training_results(log_file, run_details):
    """
    Logs the details of a training run to a CSV file.

    Parameters:
    log_file (str): The file path for the log file.
    run_details (dict): A dictionary containing details of the training run.
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=run_details.keys())

        if not file_exists:
            writer.writeheader()  # Write headers if file does not exist

        writer.writerow(run_details)


def plot_training_curves(log_file, train_loss='train_loss', val_loss='validation_loss', train_accuracy='train_accuracy', val_accuracy='validation_accuracy'):
    data = pd.read_csv(log_file)

    # Convert string representations of lists into actual lists
    data[train_loss] = data[train_loss].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data[val_loss] = data[val_loss].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data[train_accuracy] = data[train_accuracy].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data[val_accuracy] = data[val_accuracy].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Assuming the lists for each metric contain one value per epoch
    epochs = range(1, len(data['train_loss'].iloc[0]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data[train_loss].iloc[0], label='Training Loss')
    plt.plot(epochs, data[val_loss].iloc[0], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data[train_accuracy].iloc[0], label='Training Accuracy')
    plt.plot(epochs, data[val_accuracy].iloc[0], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()
    
def get_accuracy(model, data):
    """ Compute the accuracy of the `model` across a dataset `data` """
    # Ensure model is in evaluation mode, which turns off dropout
    model.eval()

    # Variables to track total and correct predictions
    correct = 0
    total = 0

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in data:
            # Get input data and labels
            inputs, labels = batch
            
            # Move data to the same device as the model
            if torch.backends.mps.is_available():
                inputs, labels = inputs.to("mps"), labels.to("mps")
            elif torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass to get outputs
            outputs, _ = model(inputs, model.init_hidden(inputs.size(0)))

            # Convert output probabilities to predicted class (0 or 1)
            predicted = outputs.round()  # Assuming a sigmoid activation at the output

            # Count total and correct predictions
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    return accuracy



print("Starting the training process...")
# Loss function
criterion = nn.BCELoss()

# Optimizer (e.g., Adam, SGD, etc.)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 5

# To store metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch
        if torch.backends.mps.is_available():
            inputs, labels = inputs.to("mps"), labels.to("mps")
        optimizer.zero_grad()
        output, _ = model(inputs, model.init_hidden(inputs.size(0)))
        # print("Output shape:", output.shape)
        # print("Labels shape:", labels.shape)

        loss = criterion(output.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Print status update every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.6f}")

    # Validation
    val_loss = 0.0
    val_accuracy = get_accuracy(model, val_loader)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            if torch.backends.mps.is_available():
                inputs, labels = inputs.to("mps"), labels.to("mps")
            output, _ = model(inputs, model.init_hidden(inputs.size(0)))
            loss = criterion(output.squeeze(1), labels.float())
            val_loss += loss.item()

    # Calculate average losses
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    # Store accuracy
    train_accuracies.append(get_accuracy(model, train_loader))
    val_accuracies.append(val_accuracy)

    # Print epoch summary
    print(f'Epoch {epoch+1} Summary')
    print(f'\tTraining Loss: {train_losses[-1]:.6f} \tTraining Accuracy: {train_accuracies[-1]:.6f}')
    print(f'\tValidation Loss: {val_losses[-1]:.6f} \tValidation Accuracy: {val_accuracies[-1]:.6f}')



run_details = {
    'epoch': epoch,
    'train_loss': train_losses,
    'validation_loss': val_losses,
    'train_accuracy': train_accuracies,
    'validation_accuracy': val_accuracies,
    'learning_rate': 0.001,  # Example hyperparameter
    # Add other hyperparameters and metrics as needed
}

print("Logging training results...")

log_training_results('training_log_partA_3_ii_v1.csv', run_details)

print(pd.read_csv('training_log_partA_3_ii_v1.csv'))

print("Plotting training curves...")

plot_training_curves('training_log_partA_3_ii_v1.csv')
