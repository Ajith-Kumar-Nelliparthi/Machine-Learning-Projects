from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder

model_file = 'mushroom_model.pkl'
with open('mushroom_model.pkl','rb') as f_in:
    model = pickle.load(f_in)
dv_file = 'mushroom_dv.pkl'
with open(dv_file,'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error':'No input data provided'}),400
    x = dv.transform([data])
    y_pred = model.predict(x)
    return jsonify({'prediction':y_pred.tolist()[0]})
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9999)