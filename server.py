from flask import Flask, request, jsonify
from sentiment import Sentiment
from AI_API.summarization import Summarizer

port = 5000

app = Flask(__name__)

# Routes

# home route
@app.route('/', methods=['GET'])
def index():
    return """ Welcome to the AI API!\n\n 
    You can use the following endpoints:\n
        - /sentiment\n
        - /summarizer\n
    """

# sentiment analysis route
@app.route('/sentiment', methods=['POST'])
def handle_sentiment_post():
    try:
        data = request.get_json()
        print(data['text'])
        sentiment = Sentiment()
        output = jsonify({
            "Success": 200,
            "request": "POST /sentiment",
            "model": data['model'],
            "data": sentiment.get_sentiment(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# summarizer route
@app.route('/summarization', methods=['POST'])
def handle_summarize_post():
    try:
        data = request.get_json()
        print(data['text'])
        summarizer = Summarizer()
        output = jsonify({
            "Success": 200,
            "request": "POST /summarization",
            "model": data['model'],
            "data": summarizer.get_summarize(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/fill_mask', methods=['POST'])
def handle_fill_mask_post():
    try:
        data = request.get_json()
        print(data['text'])
        summarizer = Summarizer()
        output = jsonify({
            "Success": 200,
            "request": "POST /fill_mask",
            "model": data['model'],
            "data": summarizer.get_summarize(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
