from flask import Flask, request, jsonify
from utils import install_model, MODEL_BASE_PATH
from pipelines.sentiment import Sentiment
from pipelines.summarizer import Summarizer
from pipelines.fill_mask import Fill_Mask
from llm import LLM

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
            "data": summarizer.summarize(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/fill_mask', methods=['POST'])
def handle_fill_mask_post():
    try:
        data = request.get_json()
        print(data['text'])
        fill_mask = Fill_Mask()
        output = jsonify({
            "Success": 200,
            "request": "POST /fill_mask",
            "model": data['model'],
            "data": fill_mask.fill_mask(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/text_generation/mistral', methods=['POST'])
def handle_text_generation_mistral_post():
    try:
        data = request.get_json()
        print(data['text'])
        output = jsonify({
            "Success": 200,
            "request": "POST /text_generation",
            "data": mistral_model.generate(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text_generation/llama', methods=['POST'])
def handle_text_generation_llama_post():
    try:
        data = request.get_json()
        output = jsonify({
            "Success": 200,
            "request": "POST /text_generation",
            "data": llama_model.generate(data["text"])
        })
        return output
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    mistral_model = Mistral(model_name="cognitivecomputations/dolphin-2.6-mistral-7b")
    llama_model = Llama(model_name="codellama/CodeLlama-7b-hf")
    llama_model.load_model()
    #mistral_model.load_model()
    app.run(debug=True, port=5000)
