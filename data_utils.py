import torch
from collections import Counter
import plotly.graph_objects as go
import plotly.io as pio
from torch import nn

# Set the default renderer for Plotly
pio.renderers.default = 'png'  # Change as needed for your environment

# def collect_tokens(data_loader):
#     all_tokens = []
#     for batch in data_loader:
#         input_ids = batch['input_ids']
#         all_tokens.extend(input_ids.flatten().tolist())
#     return all_tokens

def create_histogram(tokens, title):
    fig = go.Figure(data=[go.Histogram(x=tokens)])
    fig.update_layout(title_text=title, xaxis_title='Token ID', yaxis_title='Frequency')
    return fig

# def count_tokens(data_loader):
#     token_counts = Counter()
#     for batch in data_loader:
#         input_ids = batch['input_ids']
#         token_counts.update(input_ids.flatten().tolist())
#     return token_counts


def collect_tokens(data_loader):
    all_tokens = []
    total_batches = len(data_loader)
    for i, batch in enumerate(data_loader):
        if i % 100 == 0:  # Print progress every 100 batches
            print(f"Processing batch {i}/{total_batches} in collect_tokens")
        input_ids = batch['input_ids']
        all_tokens.extend(input_ids.flatten().tolist())
    return all_tokens

def count_tokens(data_loader):
    token_counts = Counter()
    total_batches = len(data_loader)
    for i, batch in enumerate(data_loader):
        if i % 100 == 0:  # Print progress every 100 batches
            print(f"Processing batch {i}/{total_batches} in count_tokens")
        input_ids = batch['input_ids']
        token_counts.update(input_ids.flatten().tolist())
    return token_counts





def find_token_range(data_loader):
    min_token_value = float('inf')  # Initialize with the highest possible value
    max_token_value = -float('inf') # Initialize with the lowest possible value

    total_batches = len(data_loader)
    for i, batch in enumerate(data_loader):
        if i % 100 == 0:  # Optional: Print progress every 100 batches
            print(f"Processing batch {i}/{total_batches} in find_token_range")
        input_ids = batch['input_ids']
        min_batch = torch.min(input_ids).item()  # Minimum in this batch
        max_batch = torch.max(input_ids).item()  # Maximum in this batch

        # Update global min and max
        min_token_value = min(min_token_value, min_batch)
        max_token_value = max(max_token_value, max_batch)

    return min_token_value, max_token_value

