import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Function to load the data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to preprocess the data
def preprocess_data(df):
    # Print column names to check for the correct column
    print(df.columns)

    # Convert column names to lowercase and strip any spaces
    df.columns = [col.lower().strip() for col in df.columns]

    # Ensure that the 'news' column exists
    if 'news' in df.columns:
        df['news'] = df['news'].apply(lambda x: x.lower())  # Convert text to lowercase
    else:
        raise ValueError("The 'news' column is not present in the data!")

    return df

# Function to split the data into training and testing sets
def split_data(df, test_size=0.2):
    # Use the 'news' column instead of 'text'
    X_train, X_test, y_train, y_test = train_test_split(df['news'], df['sentiment'], test_size=test_size)
    return X_train, X_test, y_train, y_test

# Function to tokenize the data using BERT tokenizer
def tokenize_data(X_train, X_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors='pt')
    X_test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors='pt')
    return X_train_enc, X_test_enc
