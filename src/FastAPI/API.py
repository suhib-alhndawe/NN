from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import re
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Initialize FastAPI application
app = FastAPI()

# Load pre-trained Keras model and Tokenizer
model = tf.keras.models.load_model('final_model.h5')  
tokenizer = Tokenizer() 

# Input model for receiving the request body
class Review(BaseModel):
    text: str

# Function to clean and preprocess the input text
def clean_text(text: str) -> str:
    """Clean the input text by removing unwanted characters."""
    text = text.lower() 
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.strip()  
    return text

# Function to predict sentiment from the cleaned text
def predict_sentiment(text: str) -> str:
    """Predict sentiment (positive, negative, or neutral) of the input text."""
    cleaned_text = clean_text(text)  # Clean the input text
    
    # Tokenize and pad the text to match input format expected by the model
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=100)  # Ensure maxlen matches the training data
    
    # Get predictions for each sentiment class
    pos_pred, neg_pred, neu_pred = model.predict(padded_seq)

    # Map the predicted sentiment index to sentiment labels
    sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
    sentiment = np.argmax([pos_pred, neg_pred, neu_pred])

    return sentiment_map[sentiment]

# Root endpoint to serve the HTML interface for sentiment analysis
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Return HTML page with text input and sentiment analysis button."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>News Sentiment Analysis</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: #ffffff;
                color: #333333;
                text-align: center;
                padding: 20px;
            }
            h1 {
                color: #4CAF50;
                font-size: 2.5em;
            }
            .container {
                max-width: 600px;
                margin: auto;
                background: #f5f5f5;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            img {
                width: 100%;
                height: auto;
                border-radius: 12px;
                margin-bottom: 15px;
            }
            textarea {
                width: 100%;
                height: 120px;
                background: #e8f5e9;
                color: #333333;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #4CAF50;
                font-size: 1em;
            }
            button {
                background: #4CAF50;
                color: white;
                padding: 12px 18px;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                background: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background: #d1f8e3;
                border-radius: 8px;
                font-size: 1.2em;
                color: #388e3c;
            }
            footer {
                margin-top: 30px;
                color: #aaaaaa;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="https://raw.githubusercontent.com/suhib-alhndawe/NN/main/image.png" alt="News Sentiment Analysis">
            <h1>Analyzing News Sentiment📊📰</h1>
            <textarea id="newsText" placeholder="Enter news text here..."></textarea>
            <button onclick="submitText()">Classify Sentiment</button>
            <div class="result" id="result"></div>
        </div>
        <footer>
            <p>by Suhib Alfurjani</p>
        </footer>
        <script>
            async function submitText() {
                const text = document.getElementById('newsText').value;
                if (text.trim() === "") {
                    alert("Please enter some text.");
                    return;
                }
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                document.getElementById('result').innerHTML = "Sentiment: " + data.sentiment;
            }
        </script>
    </body>
    </html>
    """

# Endpoint for sentiment prediction
@app.post("/predict")
async def predict(review: Review):
    """Predict sentiment of the provided review text."""
    sentiment = predict_sentiment(review.text)
    return {"sentiment": sentiment}
