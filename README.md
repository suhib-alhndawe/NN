# Sentiment Analysis using BERT 🤖

## Overview 📌
This project implements sentiment analysis using **BERT (Bidirectional Encoder Representations from Transformers)**. It processes textual data, tokenizes it with a pretrained BERT tokenizer, and trains a classification model using **PyTorch**.

## Features ✨
- **Data Preprocessing**: Cleans and prepares the text for analysis.
- **BERT Tokenization**: Uses `bert-base-uncased` tokenizer for input encoding.
- **Model Training**: Fine-tunes a **BERT for Sequence Classification** model.
- **Performance Evaluation**: Computes accuracy after each training epoch.

## Installation ⚙️
To set up the project, install the required dependencies:
```bash
pip install pandas scikit-learn transformers torch
```

## Usage 🚀
1. **Load & Preprocess Data**
   ```python
   df = load_data('data.csv')
   df = preprocess_data(df)
   ```
2. **Split Data**
   ```python
   X_train, X_test, y_train, y_test = split_data(df)
   ```
3. **Tokenization**
   ```python
   X_train_enc, X_test_enc = tokenize_data(X_train, X_test)
   ```
4. **Train the Model**
   ```python
   train_model(X_train_enc, X_test_enc, y_train, y_test, num_epochs=5)
   ```

## Model Architecture 🏗️
- **Pretrained BERT model** (`bert-base-uncased`)
- **Fully connected classification layer**
- **CrossEntropyLoss for classification**
- **Adam optimizer with a learning rate of 2e-5**

## Results 📊
The model evaluates sentiment classification accuracy after each epoch.

## Future Enhancements 🔥
- Implement **hyperparameter tuning**.
- Try **different transformer models** for better performance.
- Integrate **real-time predictions** into a web app.

## License 📜
This project is open-source under the **MIT License**.

---
🚀 Happy Coding! 🎉

