from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Preprocess text
def preprocess(text):
    text = text.lower()

    # ✅ Add here (BEFORE removing punctuation)
    text = text.replace("ai", "artificial intelligence")
    text = text.replace("ml", "machine learning")

    text = "".join([ch for ch in text if ch not in string.punctuation])
    return text

# Semantic similarity
def get_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def tfidf_similarity(t1, t2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([t1, t2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    student = request.form['student']
    model_ans = request.form['model']

    student_clean = preprocess(student)
    model_clean = preprocess(model_ans)

    # spaCy similarity
    sim1 = get_similarity(student_clean, model_clean)

# TF-IDF similarity
    sim2 = tfidf_similarity(student_clean, model_clean)

# Hybrid similarity
    similarity = (sim1 + sim2) / 2

    score = round(similarity * 10, 2)
    percentage = round(similarity * 100, 2)

    # Feedback
    if similarity > 0.8:
        feedback = "Excellent answer! Very relevant."
    elif similarity > 0.5:
        feedback = "Good answer, but can be improved."
    else:
        feedback = "Answer is not relevant. Try again."

    return render_template('index.html',
                           score=score,
                           percentage=percentage,
                           feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)