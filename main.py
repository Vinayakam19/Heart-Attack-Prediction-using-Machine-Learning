from flask import Flask, render_template ,request, jsonify, render_template
import numpy as np
import pickle
import joblib

''' Initialize the flask application '''

app = Flask(__name__)
model = pickle.load(open(r'svmbestmodel.pkl', 'rb'))

''' Deafult Page of our web application'''

@app.route("/")
def home():
    return render_template('index_test.html')

''' Page for prediction '''

@app.route("/prediction")
def prediction():
    return render_template('prediction.html')

''' Page for Login '''

@app.route("/login")
def login():
    return render_template('login.html')

''' To use the predict button in our web-app   '''

@app.route('/predict', methods =['POST'])
def predict():
 
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    output = round(prediction[0], 2)
 
    # Check the output values and retrieve the result with html tag based on the value
    if output == 1.0:
        return render_template('prediction.html',result = 'The patient is more prone to heart attack')
    else:
        return render_template('prediction.html', result = 'The patient is less prone to heart attack!')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run()