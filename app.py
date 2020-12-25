from flask import Flask, render_template, request, Markup
import joblib
import numpy as np


app = Flask(__name__)


# main index page route
@app.route('/')


def home():
    
    return render_template('index.html')




@app.route('/predict', methods=["POST", "GET"])
def predict():


    model = joblib.load('logistic_regression.pkl')


    q1 = int(request.form['q1'])
    q2 = int(request.form['q2'])
    q3 = int(request.form['q3'])
    q4 = int(request.form['q4'])
    q5 = int(request.form['q5'])
    q6 = int(request.form['q6'])
    q7 = int(request.form['q7'])
    q8 = int(request.form['q8'])
    q9 = int(request.form['q9'])
    q10 = int(request.form['q10'])


    int_features = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]


    final = [np.array(int_features)]


    prediction = model.predict_proba(final)[:,1]

    return render_template('index.html', pred = prediction)
    



if __name__ == "__main__":
    app.run(debug = True)
