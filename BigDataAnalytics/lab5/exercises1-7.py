import pandas as pd
from transformers import BertTokenizer
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dfEx = pd.DataFrame(columns=[0,1])
dfEx = dfEx._append({0:"This brilliant movie is jaw-dropping.", 1:1},
                   ignore_index=True)
dfEx = dfEx._append({0:"This movie is awful.", 1:0}, ignore_index=True)

# Show tokenized version of the example data
print("Example data tokenized:")
tokenized = dfEx[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized)

# Add padding
print("Padded example data:")
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
print(padded)

# Attention mask
print("Attention mask for example data:")
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask)

# Convert to torch tensors
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
print("Input ids tensor:")
print(input_ids)
print("Attention mask tensor:")
print(attention_mask)
