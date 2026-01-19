from flask import Flask, render_template, request
import joblib
import pandas as pd
import re

app = Flask(__name__)

# 1️⃣ Load model
model = joblib.load("model/sql_intrusion_model.pkl")

# 2️⃣ Feature extraction functions
def extract_features_from_query(query):
    query_lower = query.lower()
    return {
        'query_length': len(query),
        'token_count': len(query.split()),
        'special_char_count': len(re.findall(r"[\'\";#\-\/\*]", query)),
        'union_present': int('union' in query_lower),
        'or_present': int(' or ' in query_lower),
        'and_present': int(' and ' in query_lower),
        'drop_present': int('drop' in query_lower),
        'select_present': int('select' in query_lower),
        'tautology_present': int('or 1=1' in query_lower),
        'multiple_statements': int(';' in query),
        'comment_present': int('--' in query)
    }

def prepare_input(query):
    return pd.DataFrame([extract_features_from_query(query)])

# 3️⃣ Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    risk = None

    if request.method == "POST":
        user_query = request.form["query"]

        X_input = prepare_input(user_query)

        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        risk = round(probability * 100, 2)

        if prediction == 1:
            result = "⚠ SQL Injection Detected"
        else:
            result = "✅ Normal Query"

    return render_template("index.html", result=result, risk=risk)

# 4️⃣ Run app
if __name__ == "__main__":
    app.run(debug=True)
