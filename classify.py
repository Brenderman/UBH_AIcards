# (Concept for classify.py - You would integrate this with your generator later)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# 1. Load Data
df = pd.read_csv('data.csv')

# 2. Create the ML Pipeline
# This makes text classification simple!
model = make_pipeline(
    CountVectorizer(), # Converts text to numbers
    MultinomialNB()    # The simple classification model
)

# 3. Train the Model (In a real project, you'd use a separate test set)
model.fit(df['text'], df['category'])

# 4. Predict a generated card
new_card_text = "Your AI model won, collect prize money of 500."
prediction = model.predict([new_card_text])

print(f"The card effect is classified as: {prediction[0]}")