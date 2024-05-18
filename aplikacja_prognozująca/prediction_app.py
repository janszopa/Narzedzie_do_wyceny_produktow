from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Załaduj model
model = load('model/model.joblib')
model_columns = load('model/columns.joblib')

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
        print("Dane przed one-hot encoding:")
        print(df)
        print(model_columns)

        df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

        # Debugowanie - wyświetl dane po one-hot encoding
        print("Dane po one-hot encoding:")
        print(df)

        # Dopełnij brakujące kolumny (jeśli istnieją)
        missing_cols = set(model_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[model_columns]

        # Debugowanie - wyświetl dane przed przewidywaniem
        print("Dane przed przewidywaniem:")
        print(df)

        # Przewiduj cenę
        prediction = model.predict(df)

        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
