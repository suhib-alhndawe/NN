# 📰 News Sentiment Analysis API 🚀

## Overview
This project is a **news sentiment analysis API** that classifies news articles into **positive, negative, or neutral** sentiments using a **pre-trained deep learning model**. It provides a simple **web interface** and supports API requests for sentiment classification.

## 🔧 Technologies Used
- **Python** 🐍
- **TensorFlow/Keras** for deep learning
- **FastAPI** for API development ⚡
- **Transformers (BERT)** for tokenization
- **Docker** for containerization 🐳
- **PyTorch** for model training 🏋️
- **Scikit-learn** for data processing 📊

## 📌 Features
- Load a **pre-trained Keras model** (`final_model.h5`) for sentiment classification.
- Preprocess input text (cleaning, tokenization, and padding).
- Provide a **FastAPI endpoint** (`/predict`) for sentiment analysis.
- Serve a **user-friendly web interface** for text input and classification.
- Deployable with **Docker** for scalability and portability.
- Train a **BERT-based model** for sentiment classification.

## 🛠 Installation & Usage
### 1️⃣ Clone the repository:
```bash
 git clone https://github.com/your-repo/news-sentiment-api.git
 cd news-sentiment-api
```
### 2️⃣ Run with Docker:
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```
### 3️⃣ Access the web interface:
Open your browser and go to: 
```
http://localhost:8000/
```
### 4️⃣ API Usage (via cURL):
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "The economy is improving rapidly!"}'
```

## 📊 Example Output:
```json
{
  "sentiment": "positive"
}
```

## 🏋️ Model Training
### 1️⃣ Load and Preprocess Data:
- Load dataset from CSV.
- Convert text to lowercase and clean it.
- Tokenize using **BERT tokenizer**.

### 2️⃣ Split Data:
- Use `train_test_split` to split data into **training** and **testing** sets.

### 3️⃣ Encode and Prepare Data:
- Convert text into **BERT tokenized format**.
- Encode sentiment labels using **LabelEncoder**.

### 4️⃣ Train Model:
- Load **BERT for sequence classification**.
- Train for a defined number of **epochs**.
- Use **CrossEntropyLoss** and **Adam optimizer**.

### 5️⃣ Evaluate Model:
- Compute **accuracy** on the test dataset.
- Print training progress and results.

## 🌐 **Deployment on Docker Hub**
I have deployed the model on **Docker Hub** for easy access and distribution. You can pull the Docker image from the following link:

- **Docker Hub Repository**: [suhibalfurjani/news](https://hub.docker.com/repository/docker/suhibalfurjani/news)

## 📌 Author
Developed by **Suhib Alfurjani** 🚀
