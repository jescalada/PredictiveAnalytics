import numpy as np
import pandas as pd

import torch
import transformers as ppb
import warnings
import os

warnings.filterwarnings('ignore')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
FILE = "movie_reviewsBERT.csv"
batch_1 = pd.read_csv(PATH + FILE, delimiter=',', header=None)

print(batch_1.shape)
ROW = 1
print("Review 1st column: " + batch_1.iloc[ROW][0])
print("Rating 2nd column: " + str(batch_1.iloc[ROW][1]))

# Show counts for review scores.
print("** Showing review counts")
print(batch_1[1].value_counts())

# Load pretrained models.
# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                    ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Tokenize the sentences.
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print("\n****************** Tokenized reviews ")
print(tokenized)
print(tokenized.values)
print("******************")

# For processing we convert to 2D array.
max_len = 0

# Get maximum number of tokens (get biggest sentence).
print("\nGetting maximum number of tokens in a sentence")
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

print("Most tokens in a review (max_len): " + str(max_len))

# Add padding
print("------------")
print("Padded so review arrays as same size: ")
padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
print("These are the padded reviews:")
print(padded)
print("This is the last padded sentence:")
LAST_INDEX = len(batch_1) - 1
print(padded[LAST_INDEX])
print("\n------------")
print("Attention mask tells BERT to ignore the padding.")

# Sending padded data to BERT would slightly confuse it
# so create a mask to tell it to ignore the padding.
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask.shape)
print(attention_mask)
print(attention_mask[LAST_INDEX])
print("=============")

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
print("Input ids which are padded reviews in torch tensor format:")
print(input_ids)
print("Attention mask in torch tensor format:")
print(attention_mask)
print("++++++++++++++")

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
print(features[1999])
print("-------------------------")