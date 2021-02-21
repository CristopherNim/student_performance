from flask import Flask, request, jsonify, render_template, session, flash, redirect, url_for, g
import os
import pickle
from ml_model import row_pred

app = Flask('app')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def test():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features = [x for x in request.form.values()]
    predictions = row_pred(int_features)

    return render_template('index.html', prediction_text='predicted student math score is: {}'.format(predictions))


@app.route('/results', methods=['GET', "POST"])
def results():
    data = request.get_json(force=True)
    prediction = row_pred(list(data.values()))
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)


