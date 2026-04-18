from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

application = Flask(__name__)

# Train the model on startup (avoids joblib version mismatch)
texts = [
    "I love this movie, it is fantastic!",
    "Absolutely terrible, worst experience ever.",
    "Great acting and wonderful plot.",
    "I hated every minute of it.",
    "A beautiful and inspiring story.",
    "Boring, dull, and a waste of time."
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided. Please send a JSON with a "text" key.'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction,
        'model_version': '1.0'
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
