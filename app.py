print("App.py started")  # Debug
import os
print("best_model.pkl exists:", os.path.exists('best_model.pkl'))
print("scaler.pkl exists:", os.path.exists('scaler.pkl'))

import pickle

print("Loading model...")
model = pickle.load(open('best_model.pkl', 'rb'))
print("Model loaded.")

print("Loading scaler...")
scaler = pickle.load(open('scaler.pkl', 'rb'))
print("Scaler loaded.")

from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil input dari form
            data = request.form['data']
            # Ubah ke array numpy
            input_data = np.array([float(i) for i in data.split(',')]).reshape(1, -1)
            # Scaling
            input_scaled = scaler.transform(input_data)
            # Prediksi
            prediction = model.predict(input_scaled)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)