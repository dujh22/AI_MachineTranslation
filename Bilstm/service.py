from flask import Flask, render_template, request, jsonify
from train_test import translate


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translation', methods=['POST'])
def translation():
    text = request.json['text']
    text = text.strip()
    translated_text = translate(text)  
    return jsonify({'translatedText': translated_text})

if __name__ == '__main__':
    app.run()