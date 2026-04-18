from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)

# Load the trained model into memory
model = joblib.load('sentiment_model.joblib')

@application.route('/predict', methods=['POST'])
def predict():
    # Parse the incoming JSON request
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided. Please send a JSON with a "text" key.'}), 400

    # Make a prediction
    prediction = model.predict([text])[0]

    # Return the result
    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction,
        'model_version': '1.0'
    })

if __name__ == '__main__':
    # Listen on all network interfaces so Docker/Beanstalk can expose it
    application.run(host='0.0.0.0', port=5000)
