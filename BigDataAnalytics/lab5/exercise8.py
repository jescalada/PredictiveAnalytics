import pandas as pd
import numpy as np
import torch
import transformers as ppb

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                    ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

df_ex = pd.DataFrame(columns=[0, 1])
df_ex = df_ex._append({0: "This brilliant movie is jaw-dropping.", 1: 1},
                      ignore_index=True)
df_ex = df_ex._append({0: "This movie is awful.", 1: 0}, ignore_index=True)

# Show tokenized version of the example data
print("Example data tokenized:")
tokenized = df_ex[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
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

# The model() function runs our sentences through BERT. The results of the
# processing will be returned into last_hidden_states.
print("BERT model transforms tokens and attention mask tensors into features ")
print("for logistic regression.")
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# Save the features and these will become
# features of the logistic regression model.
features = last_hidden_states[0][:, 0, :].numpy()

print("Let's see the features: ")
print(features)
print(features[0])
print("-------------------------")
