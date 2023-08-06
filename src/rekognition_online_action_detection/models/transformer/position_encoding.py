# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)     

    def forward(self, x, padding=0):
        x = x + self.pe[padding: padding + x.shape[0], :]
        return self.dropout(x)


# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(RelativePositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.d_model = d_model
#         self.max_len = max_len

#         relative_position_matrix = torch.zeros(2 * max_len, d_model)

#         # Initialize the relative position matrix using sine and cosine functions
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         sinusoid_term = position * div_term
#         relative_position_matrix[:max_len, 0::2] = torch.sin(sinusoid_term)
#         relative_position_matrix[:max_len, 1::2] = torch.cos(sinusoid_term)
#         relative_position_matrix[max_len:, 0::2] = torch.sin(sinusoid_term)
#         relative_position_matrix[max_len:, 1::2] = torch.cos(sinusoid_term)
#         print(f"RPE dim: {d_model}")
#         relative_position_matrix = relative_position_matrix.unsqueeze(0).transpose(0, 1)
#         print(f"PE dim: {relative_position_matrix.shape}")
#         self.register_buffer('pe', relative_position_matrix) 
        

#     def forward(self, x, padding=0):        
#         # print(f"Input shape: {x.shape}, Padding: {padding}")
#         seq_len = x.shape[0]
#         relative_position_matrix_slice = self.pe[self.max_len - seq_len + padding:self.max_len + padding, :]
#         x = x + relative_position_matrix_slice
#         return self.dropout(x)



# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, max_position, embedding_dim):
#         super(RelativePositionalEncoding, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.positional_embeddings = nn.Embedding(2 * max_position - 1, embedding_dim)
#         self.max_position = max_position

#     def forward(self, input_seq):
#         seq_length = input_seq.size(1)
#         positions = torch.arange(seq_length, device=input_seq.device)
#         relative_positions = torch.unsqueeze(positions, 0) - torch.unsqueeze(positions, 1) + self.max_position - 1
#         relative_positional_encodings = self.positional_embeddings(relative_positions)
#         return input_seq + relative_positional_encodings

# # Example usage
# max_position = 100
# embedding_dim = 256
# seq_length = 50
# batch_size = 16

# input_data = torch.randn(batch_size, seq_length, embedding_dim)
# positional_encoder = RelativePositionalEncoding(max_position, embedding_dim)
# output_data = positional_encoder(input_data)