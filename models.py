from transformers import BertModel
import torch
import torch.nn as nn

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



class SentimentClassifierPooled(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifierPooled, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

        # Linear layer to map the pooled output to the number of classes
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Getting pooled output from BERT
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # Applying dropout
        output = self.dropout(pooled_output)

        # Passing through the linear layer
        return self.out(output)


class SentimentClassifierLast(nn.Module):
    def __init__(self, pre_trained_model_name):
        super(SentimentClassifierLast, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

        # Linear layer to map the last hidden state of the [CLS] token to a single output
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Print device of input tensors
        print("Input device:", input_ids.device)

        # Getting the last hidden state output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Print device of the output from the embedding layer
        print("Embedding output device:", last_hidden_state.device)

        # Extract the [CLS] token's embeddings
        cls_token_embedding = last_hidden_state[:, 0, :]

        # Applying dropout
        dropout_output = self.dropout(cls_token_embedding)
        print("Dropout output device:", dropout_output.device)  # Check device after dropout
        # Passing through the linear layer and applying sigmoid
        
        final_output = self.out(dropout_output).squeeze()
        print("Final output device before sigmoid:", final_output.device)  # Check device before sigmoid

        return torch.sigmoid(final_output)
        



# class SentimentClassifierLast(nn.Module):
#     def __init__(self, pre_trained_model_name):
#         super(SentimentClassifierLast, self).__init__()
#         self.bert = BertModel.from_pretrained(pre_trained_model_name)

#         # Dropout layer to prevent overfitting
#         self.dropout = nn.Dropout(p=0.3)

#         # Linear layer to map the last hidden state of the [CLS] token to a single output
#         self.out = nn.Linear(self.bert.config.hidden_size, 1)

#     def forward(self, input_ids, attention_mask):
#         # Getting the last hidden state output
#         last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

#         # Extract the [CLS] token's embeddings
#         cls_token_embedding = last_hidden_state[:, 0, :]

#         # Applying dropout
#         dropout_output = self.dropout(cls_token_embedding)

#         # Passing through the linear layer and applying sigmoid
#         return torch.sigmoid(self.out(dropout_output))
        
