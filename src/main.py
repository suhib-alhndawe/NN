from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pandas as pd
import torch
from model_training import train_model

# Load the dataset
df = pd.read_csv('news.csv')

# Split the dataset into features (X) and labels (y)
X = df['news']  
y = df['sentiment'] 

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data for training and testing using BERT's tokenizer
X_train_enc = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
X_test_enc = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

# Train the model using the training data and evaluate it on the test data
train_model(X_train_enc, X_test_enc, y_train, y_test)
