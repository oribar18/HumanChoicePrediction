import os
import warnings
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

warnings.filterwarnings("ignore")

# Directory containing the CSV files
data_dir = 'data/game_reviews'

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def preprocess_review(review):
    inputs = tokenizer(review, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return inputs

def predict_sentiment(review):
    inputs = preprocess_review(review)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    confidence_score = np.max(probabilities)
    predicted_class = np.argmax(probabilities)
    # Combine very negative and negative into negative, and positive and very positive into positive
    if predicted_class in [0, 1]:  # Very negative or negative
        sentiment = 'negative'
    elif predicted_class in [3, 4]:  # Positive or very positive
        sentiment = 'positive'
    else:  # Neutral
        # Consider neutral as negative or positive based on your preference
        sentiment = 'negative'
    return sentiment, confidence_score

# Function to process a single CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    results = []
    for _, row in df.iterrows():
        review_pt1 = str(row[2]) if not pd.isna(row[2]) else ""
        review_pt2 = str(row[3]) if not pd.isna(row[3]) else ""
        review_text = (review_pt1 + " " + review_pt2).strip()
        review_text = review_text.replace('\n', ' ').replace('\r', ' ')  # Replace newline characters
        if review_text:  # Skip if the concatenated review is empty
            sentiment, confidence = predict_sentiment(review_text)
            results.append((row[0], row[1], review_text, sentiment, confidence))
    return results

# Process all CSV files and save the results
all_results = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        results = process_csv(file_path)
        all_results.extend(results)

# Save the results to a text file
with open('sentiment_confidence_results.txt', 'w', encoding='utf-8') as file:
    for review_id, hotel_name, review, sentiment, confidence in all_results:
        file.write(f"{review_id}\t{hotel_name}\t{review}\t{sentiment}\t{confidence:.2f}\n")

print("Sentiment analysis and confidence scoring completed.")