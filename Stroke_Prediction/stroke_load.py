from flask import Flask, request, jsonify
import pandas as pd
import pickle

model_file = 'stroke_predict.pkl'
with open('stroke_predict.pkl', 'rb') as f_in:
    brf = pickle.load(f_in)

dv_filename = 'stroke_health_dv.pkl'
with open(dv_filename, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    x = pd.DataFrame([data])
    x = dv.transform([data])
    y_pred = brf.predict(x)

    result = {
            'stroke_probability': float(y_pred[0])
        }
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
