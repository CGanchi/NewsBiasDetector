from barebones_model import BareBonesLinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from joblib import dump

# Load and prepare data
dataFrame = pd.read_csv('Final_Trimmed_Bias_Ratings_Dataset.csv')
dataFrame["text"] = dataFrame["text"].fillna("")
dataFrame["title"] = dataFrame["title"].fillna("")

# Combine title and text for better bias detection
dataFrame["combined_text"] = dataFrame["title"] + " " + dataFrame["text"]

y = dataFrame["bias_rating"]

# Checking on Dataset
print("--- Bias Spread Analysis ---")
print(f"Bias rating range: {y.min():.3f} to {y.max():.3f}")
print(f"Bias rating mean: {y.mean():.3f}")
print(f"Bias rating std: {y.std():.3f}")
print(f"Sample size: {len(y)}")
print(f"Unique values: {len(y.unique())}")
print(f"Left 10 ratings: {sorted(y.unique())[:10]}")
print()

# Split data into training and testing using combined text
X_train, X_test, y_train, y_test = train_test_split(
    dataFrame["combined_text"], y, test_size=0.2, random_state=42
)

# Vectorize text with more focused parameters for bias detection
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',   # get rid of common words
    max_features=5000,   # Reduced for better focus on important words
    min_df= 3,           # Must appear in at least 3 documents
    max_df= 0.7,         # Ignore very common words (appear in 70% of articles)
    ngram_range=(1, 2), # Include both single words and bigrams
    sublinear_tf=True   # Use log scaling for TF
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train model
print("Training model...")
model = BareBonesLinearRegression()

# Higher learning rate and more epochs for stronger learning
model.fit(X_train_tfidf, y_train, learning_rate=0.01, epochs=250000, verbose=True)


# store model and vectorizer
dump(model, 'bias_model.joblib')
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("\n--- Training Complete ---")

'''

# Look at training progress

rint(f"Training cost - Start: {model.costs[0]:.6f}, End: {model.costs[-1]:.6f}")

# Make predictions
predictions = model.predict(X_test_tfidf)
print(f"Predictions range: {predictions.min():.3f} to {predictions.max():.3f}")
print(f"Predictions std: {predictions.std():.6f}")

# Look at training progress

rint(f"Training cost - Start: {model.costs[0]:.6f}, End: {model.costs[-1]:.6f}")

# Make predictions
predictions = model.predict(X_test_tfidf)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.4f}")

# Baseline comparison (predicting mean)
baseline_pred = np.full_like(predictions, np.mean(y_train))
print(f"Baseline MAE (predicting mean): {mean_absolute_error(y_test, baseline_pred):.4f}")

'''
