import torch
import torch.optim as optim
from torch import nn
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


def train_model_rnn(model, train_loader, val_loader, epochs=5, learning_rate=0.001, weight_decay=1e-5, use_clipping=False, clip_value=1.0):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Record the start time
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs, labels = batch['review'], batch['label']
            if torch.backends.mps.is_available():
                inputs, labels = inputs.to("mps"), labels.to("mps")
            optimizer.zero_grad()
            output, _ = model(inputs, model.init_hidden(inputs.size(0)))
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item()

            # Progress update
            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                total_batches = epochs * len(train_loader)
                completed_batches = epoch * len(train_loader) + i + 1
                estimated_total_time = elapsed_time / completed_batches * total_batches
                remaining_time = estimated_total_time - elapsed_time
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.6f}")
                print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

        # Validation phase
        val_loss, val_accuracy = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['review'], batch['label']
                if torch.backends.mps.is_available():
                    inputs, labels = inputs.to("mps"), labels.to("mps")
                output, _ = model(inputs, model.init_hidden(inputs.size(0)))
                loss = criterion(output.squeeze(), labels.float())
                val_loss += loss.item()
                preds = output.round().squeeze()
                val_accuracy += torch.sum(preds == labels).item()

        # Calculate average losses and accuracies
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(get_accuracy(model, train_loader, is_rnn=True))
        val_accuracies.append(val_accuracy / len(val_loader))

        # Print epoch summary
        print(f'Epoch {epoch+1} Summary')
        print(f'\tTraining Loss: {train_losses[-1]:.6f} \tTraining Accuracy: {train_accuracies[-1]:.6f}')
        print(f'\tValidation Loss: {val_losses[-1]:.6f} \tValidation Accuracy: {val_accuracies[-1]:.6f}')

    # Record the end time and calculate duration
    end_time = time.time()
    duration = str(timedelta(seconds=end_time - start_time))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'timestamp': timestamp,
        'duration': duration
    }


def train_model(model, train_loader, val_loader, criterion_func, epochs=5, learning_rate=0.001, weight_decay=1e-5, use_clipping=False, clip_value=1.0):
    criterion = criterion_func()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['targets']

            print(f"Batch {i+1} - Original device: inputs {input_ids.device}, labels {labels.device}")

            if torch.backends.mps.is_available():
                input_ids, attention_mask, labels = input_ids.to("mps"), attention_mask.to("mps"), labels.to("mps")

            print(f"Batch {i+1} - After moving: inputs {input_ids.device}, labels {labels.device}")
            
            optimizer.zero_grad()

            # Adjust the model's forward pass as per its implementation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            print(f"Batch {i+1} - Model outputs device: {outputs.device}")

            # loss = criterion(outputs, labels.float())
            loss = criterion(outputs.squeeze(), labels.float())
            print("Loss tensor device:", loss.device)  # Check device of loss tensor


            loss.backward()

            # Apply gradient clipping if enabled
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.6f}")

        # Validation phase
        val_loss = 0.0
        val_accuracy = get_accuracy(model, val_loader)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['targets']
                if torch.backends.mps.is_available():
                    input_ids, attention_mask, labels = input_ids.to("mps"), attention_mask.to("mps"), labels.to("mps")

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

        # Record and report metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(get_accuracy(model, train_loader))
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}, Training Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}')

    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(timedelta(seconds=duration))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Return training results
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'timestamp': timestamp,
        'duration': formatted_duration
    }


def get_accuracy(model, data_loader, is_rnn=False):
    """ Compute the accuracy of the `model` across a dataset `data_loader` """
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in data_loader:
            if is_rnn:
                # RNN expects inputs in a different format
                inputs, labels = batch['review'], batch['label']
                outputs, _ = model(inputs, model.init_hidden(inputs.size(0)))
                preds = outputs.round().squeeze()  # Assuming binary classification with sigmoid
            else:
                # Handle models expecting BERT-style inputs
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['targets']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)  # For multi-class classification

            # Move data to the correct device
            if torch.backends.mps.is_available():
                labels = labels.to("mps")
            elif torch.cuda.is_available():
                labels = labels.cuda()

            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    accuracy = correct_predictions.double() / total_predictions
    return accuracy.item()



# def get_accuracy(model, data_loader):
#     """ Compute the accuracy of the `model` across a dataset `data_loader` """
#     model.eval()  # Set the model to evaluation mode

#     correct_predictions = 0
#     total_predictions = 0

#     with torch.no_grad():  # Disable gradient calculations
#         for batch in data_loader:
#             input_ids = batch['input_ids']
#             attention_mask = batch['attention_mask']
#             targets = batch['targets']

#             # Get model predictions
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)

#             # Get the predicted class (the one with the highest probability)
#             preds = torch.argmax(outputs, dim=1)

#             # Count correct predictions
#             correct_predictions += torch.sum(preds == targets)
#             total_predictions += targets.size(0)

#     # Calculate accuracy
#     accuracy = correct_predictions.double() / total_predictions
#     return accuracy.item()


def log_training_results(filename, details):
    with open(filename, 'w') as f:
        f.write(','.join(details.keys()) + '\n')
        f.write(','.join(map(str, details.values())))

def plot_training_curves(log_file):
    data = pd.read_csv(log_file)

    epochs = range(1, len(data['train_losses']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data['train_losses'], label='Training Loss')
    plt.plot(epochs, data['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data['train_accuracies'], label='Training Accuracy')
    plt.plot(epochs, data['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()
