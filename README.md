
---

# News Sentiment Analysis with BERT 📰🤖

This project uses BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis of news articles. It preprocesses the data, tokenizes it using the BERT tokenizer, and trains a sequence classification model using the pre-trained BERT model.

## 📂 Project Structure

```
├── data/                     # Data files (CSV format)
├── notebook/                 # Jupyter notebook for analysis (optional)
├── src/                      # Source code (Python scripts)
│   ├── load_data.py          # Load and preprocess the data
│   ├── preprocess.py         # Text preprocessing
│   ├── model.py              # Model training and evaluation
│   └── utils.py              # Helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the data and place it in the `data/` folder. You should have a CSV file with at least two columns:
   - `news`: The text of the news article.
   - `sentiment`: The sentiment label (e.g., `positive`, `negative`).

## 🧑‍💻 Usage

### 1. Load and Preprocess Data

To load and preprocess the dataset, you can use the provided functions in the script:

```python
from src.load_data import load_data
from src.preprocess import preprocess_data

file_path = 'data/news_data.csv'
df = load_data(file_path)
df = preprocess_data(df)
```

### 2. Split the Data

Next, split the data into training and testing sets:

```python
from src.split_data import split_data

X_train, X_test, y_train, y_test = split_data(df)
```

### 3. Tokenize the Data

Tokenize the text data using the BERT tokenizer:

```python
from src.tokenize_data import tokenize_data

X_train_enc, X_test_enc = tokenize_data(X_train, X_test)
```

### 4. Train the Model

Train the BERT model on the preprocessed and tokenized data:

```python
from src.model import train_model

train_model(X_train_enc, X_test_enc, y_train, y_test, num_epochs=5)
```

The model will be trained for 5 epochs, and accuracy will be displayed for each epoch.

## 🛠️ Requirements

To run this project, you need the following Python libraries:

- `pandas`
- `sklearn`
- `transformers`
- `torch`

You can install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 📈 Results

The model is trained using BERT, and the results are evaluated based on accuracy. After training, you will see the accuracy for each epoch printed in the console.

## 📝 Notes

- The dataset must include a column named `news` for the news articles and a `sentiment` column for labels.
- The code uses `BERT-base-uncased` for tokenization and model training.
- This implementation assumes a binary classification task (2 sentiment labels). If you have more labels, adjust the number of labels in the BERT model.

## 🧑‍💻 Contributors

- **suhib-alhndawe** - [GitHub Profile]([https://github.com/suhib-alhndawe](https://github.com/suhib-alhndawe)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

