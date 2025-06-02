from barebones_model import BareBonesLinearRegression
from joblib import load

# load in the vectorizer and model from newsBias.py after training
model = load('bias_model.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')


# Function to predict bias rating for a given article text using the above model and vectorizer
def predict_bias_rating(article_text, vectorizer, trained_model):
    # Transform text
    article_tfidf = vectorizer.transform([article_text])
    
    # Get prediction
    prediction = trained_model.predict(article_tfidf)[0]
    

    return float(prediction)

print(predict_bias_rating("text", tfidf_vectorizer, model))
