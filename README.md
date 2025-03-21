# 📰 News Sentiment Analysis API 🚀

## Overview
This project is a **news sentiment analysis API** that classifies news articles into **positive, negative, or neutral** sentiments using a **pre-trained deep learning model**. It provides a simple **web interface** and supports API requests for sentiment classification.

## 🔧 Technologies Used
- **Python** 🐍
- **TensorFlow/Keras** for deep learning
- **FastAPI** for API development ⚡
- **Transformers (BERT)** for tokenization
- **Docker** for containerization 🐳

## 📌 Features
- Load a **pre-trained Keras model** (`final_model.h5`) for sentiment classification.
- Preprocess input text (cleaning, tokenization, and padding).
- Provide a **FastAPI endpoint** (`/predict`) for sentiment analysis.
- Serve a **user-friendly web interface** for text input and classification.
- Deployable with **Docker** for scalability and portability.

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

## 📌 Author
Developed by **Suhib Alfurjani** 🚀

---
Feel free to contribute or provide feedback! ✨

