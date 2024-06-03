import pandas as pd

from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)

# Załaduj model
model = load('model/model.joblib')
model_columns = load('model/columns.joblib')
scaler = load('model/scaler.joblib')

# Strona główna
@app.route('/')
def home():
    return render_template('index.html')


# Endpoint do przewidywania ceny
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobierz dane z formularza
        data = request.form.to_dict()

        # Przekształć dane na odpowiedni format
        df = pd.DataFrame([data])
        # Debugowanie - wyświetl dane przed one-hot encoding

        df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

        # Dopełnij brakujące kolumny (jeśli istnieją)
        categorical_features = ['cut', 'color', 'clarity']
        missing_cols = set(model_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[model_columns]

        # Przewiduj cenę
        df = scaler.transform(df)
        print(df)
        prediction = model.predict(df) #TODO różne typy danych model wytrenowany na np.ndarray bez nazw a tu DataFrame z nazwami kolumn

        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
