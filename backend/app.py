import pandas as pd
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

print("Thorough data cleaning completed. Cleaned files saved as 'Cleaned_True.csv' and 'Cleaned_Fake.csv'.")
