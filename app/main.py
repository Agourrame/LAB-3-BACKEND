from flask import Flask, request, jsonify
import pickle
import numpy as np
from keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models and scaler
student_model = pickle.load(open('educational_system/models/student_model.pkl', 'rb'))
scaler = pickle.load(open('educational_system/models/scaler.pkl', 'rb'))
emotion_model = load_model('educational_system/models/emotion_model.h5')

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict_student', methods=['POST'])
def predict_student():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)  # Apply the same scaling as during training
    prediction = student_model.predict(features)
    response = jsonify(prediction=int(prediction[0]))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('L').resize((48, 48))
    image = np.array(image).reshape(1, 48, 48, 1) / 255.0
    prediction = emotion_model.predict(image)
    response = jsonify(prediction=prediction.tolist())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
