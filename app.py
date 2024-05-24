import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask, request, jsonify

nltk.download('punkt')
nltk.download('stopwords')
app = Flask(__name__)

df = pd.read_csv("published_courses_update_new.csv")
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text)
    return [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

def recommend_courses(keyword, num_recommendations=5):
    keyword_vector = tfidf_vectorizer.transform([keyword])
    cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    related_course_indices = cosine_similarities.argsort()[::-1][:num_recommendations]
    recommended_courses = df.iloc[related_course_indices][['Course Title','url']]
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
    recommended_courses = recommend_courses('yoga')
    response = format_response(recommended_courses)
    return jsonify(response)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    res = process_request(req)
    res = jsonify(res)
    return res

def process_request(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    skill = parameters.get("stringinput")
    intent = result.get("intent").get('displayName')

    if intent == 'Recommendation System':
        recommended_courses = recommend_courses(skill)
        response = format_response(recommended_courses)
        return response

if __name__ == '__main__':
    app.run(debug=True)
