import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for handling cross-origin requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Load stop words and initialize lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions, hashtags, special characters, and numbers
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase and strip extra whitespace
    text = text.lower().strip()
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2]
    return " ".join(words)

def standardize_date(date_text):
    # Standardize date format if applicable
    try:
        date = datetime.strptime(date_text, '%B %d, %Y')  # example format: "December 19, 2016"
        return date.strftime('%Y-%m-%d')
    except ValueError:
        return date_text  # Return as-is if parsing fails

def clean_dataframe(df):
    # Drop duplicates and rows with null text or title
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['text', 'title'], inplace=True)
    
    # Clean text and title columns
    df['text'] = df['text'].apply(clean_text)
    df['title'] = df['title'].apply(clean_text)
    
    # Standardize date formats
    if 'date' in df.columns:
        df['date'] = df['date'].apply(standardize_date)
    
    # Remove articles with short text (e.g., fewer than 50 words) if needed
    df = df[df['text'].str.split().apply(len) > 50]
    return df

# Clean both dataframes
clean_true_df = clean_dataframe(true_df)
clean_fake_df = clean_dataframe(fake_df)

# Save cleaned dataframes to new CSV files
clean_true_df.to_csv("Cleaned_True.csv", index=False)
clean_fake_df.to_csv("Cleaned_Fake.csv", index=False)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Load cleaned datasets
true_data = pd.read_csv("Cleaned_True.csv")
fake_data = pd.read_csv("Cleaned_Fake.csv")

# Add labels: 0 for True News, 1 for Fake News
true_data['label'] = 0
fake_data['label'] = 1

# Combine datasets
data = pd.concat([true_data, fake_data]).reset_index(drop=True)

# Select relevant columns
data = data[['text', 'label']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load the saved model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Prediction function
def predict_news(text):
    text_tfidf = vectorizer.transform([text])  # Transform input text to TF-IDF features
    prediction = model.predict(text_tfidf)[0]
    confidence = max(model.predict_proba(text_tfidf)[0])  # Get confidence score
    result = "Fake News" if prediction == 1 else "True News"
    return {"result": result, "confidence": confidence}

# Flask API for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Get prediction
        prediction = predict_news(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
