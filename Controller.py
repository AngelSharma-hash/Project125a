from flask import Flask, jsonify, request
app = Flask(__name__)
from Project import get_Prediction 

@app.route('/predict_alphabet', methods = ['POST'])

def Predict_Data():
    image = request.files.get("alphabet")
    prediction = get_Prediction(image)
    return jsonify({
        "prediction":prediction
    }),200
if __name__ == '__main__':
    app.run(debug = True)
