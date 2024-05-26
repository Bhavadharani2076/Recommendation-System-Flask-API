import numpy as np
from flask import Flask, request, make_response
import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the courses data
df = pd.read_csv("published_courses_update_new.csv")
stop_words = set(stopwords.words('english'))

# Define the tokenizer function
def tokenize(text):
    tokens = word_tokenize(text)
    return [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

# Initialize the TF-IDF vectorizer and fit it on the course data
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Function to recommend courses
def recommend_courses(keyword, num_recommendations=5):
    keyword_vector = tfidf_vectorizer.transform([keyword])
    cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    related_course_indices = cosine_similarities.argsort()[::-1][:num_recommendations]
    recommended_courses = df.iloc[related_course_indices][['Course Title', 'url']]
    return recommended_courses

def format_response(recommended_courses):
    course_titles = recommended_courses['Course Title'].tolist()
    course_urls = recommended_courses['url'].tolist()
    
    response = {
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [
                        "The courses that match your requirements are:"
                    ]
                }
            },
            {
                "payload": {
                    "richContent": [
                        [
                            {
                                "type": "chips",
                                "options": [{"text": title, "link": url} for title, url in zip(course_titles, course_urls)]
                            }
                        ]
                    ]
                }
            }
        ]
    }
    return response

@app.route('/')
def hello():
    return 'Hello World. I am UplyrnBot.'

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r  # Final Response sent to DialogFlow

def processRequest(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    intent = result.get("intent").get('displayName')
    
    if intent == 'Recommendation System':
        # Check if the user has already provided a skill
        skill = parameters.get("skill")
        if not skill:
            # If skill is not provided, prompt the user for input
            fulfillmentText = "Sure! What skill or topic are you interested in? For example, you can say 'programming' or 'data science'."
            return {"fulfillmentText": fulfillmentText}
        else:
            # If skill is provided, generate recommendations based on the skill
            recommended_courses = recommend_courses(skill)
            response = format_response(recommended_courses)
            return response


if __name__ == '__main__':
    app.run(debug=True)
