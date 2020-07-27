from flask import Flask, render_template, request
import pickle
import numpy as np
import constants
import pandas as pd


# load the model from disk
saved_model = constants.SAVED_MODEL
loaded_model = pickle.load(open(saved_model, 'rb'))


# Flask app:
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        batting_team = request.form['batting-team']
        bowling_team = request.form['bowling-team']             
        overs = float(request.form['overs'])
        runs = request.form['runs']
        wickets = request.form['wickets']
        runs_in_prev_5 = request.form['runs_in_prev_5']
        wickets_in_prev_5 = request.form['wickets_in_prev_5']

        features_array = [batting_team,bowling_team,runs,wickets,overs,runs_in_prev_5,wickets_in_prev_5]

        data = pd.DataFrame(features_array).T
        data.columns = constants.FEATURES_REQUIRED
        data = pd.get_dummies(data = data, columns = ['bat_team','bowl_team'])
        for i in constants.REQUIRED_FEATURES_FORMAT:
            if i not in data.columns.unique():
                data[i] = 0
        data.runs = int(data.runs)
        data.wickets = int(data.wickets)
        data.overs = float(data.overs)
        data.runs_last_5 = int(data.runs_last_5)
        data.wickets_last_5 = int(data.wickets_last_5)
        data = data[constants.REQUIRED_FEATURES_FORMAT]

        my_prediction = int(loaded_model.predict(data)[0])

        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)


if __name__ == '__main__':
	app.run(debug=True)






