import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feat_list = request.form.to_dict()
    feat_list = list(feat_list.values())
    feat_list = list(map(int, feat_list))
    print(feat_list)
    feat_list = np.array(feat_list).reshape(1,17)
    print("\n",feat_list)
    prediction = model.predict(feat_list)
    out  = prediction[0]
    if out == 0:
        text = "No patient does not have  heart disease"
    else:
        text = "Yes patient has a heart disease"
    return render_template('index.html', Prediction_text='Result : {}'.format(text))

if __name__ =='__main__':
    app.run(debug=True)