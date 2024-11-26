import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

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
data = data[['text', 'label']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load the model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Prediction function
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = max(model.predict_proba(text_tfidf)[0])
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

        prediction = predict_news(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
