from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import sqlite3

app = Flask(__name__)

# =============================
# Load ML model
# =============================
model = joblib.load("model/sql_intrusion_model.pkl")

# =============================
# Constants
# =============================
ALLOWED_TABLES = {"users", "products", "orders", "payments", "logs"}
RISK_THRESHOLD = 0.6  # 60%

# =============================
# Database helper
# =============================
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# =============================
# Feature extraction (ML)
# =============================
def extract_features_from_query(query):
    query_lower = query.lower()

    return {
        "query_length": len(query),
        "token_count": len(query.split()),
        "special_char_count": len(re.findall(r"[\'\";#\-\/\*]", query)),
        "union_present": int("union" in query_lower),
        "or_present": int(" or " in query_lower),
        "and_present": int(" and " in query_lower),
        "drop_present": int("drop" in query_lower),
        "select_present": int("select" in query_lower),
        "tautology_present": int("or 1=1" in query_lower),
        "multiple_statements": int(";" in query),
        "comment_present": int("--" in query),
    }

def prepare_input(query):
    return pd.DataFrame([extract_features_from_query(query)])

# =============================
# SQL table extraction
# =============================
def extract_tables_from_query(query):
    query = query.lower()
    tables = re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", query)
    return set(tables)

def is_table_allowed(query):
    tables = extract_tables_from_query(query)

    if not tables:
        return False, tables

    for table in tables:
        if table not in ALLOWED_TABLES:
            return False, tables

    return True, tables

# =============================
# Routes
# =============================
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
            result = "✅ Normal Query Detected"

    return render_template("index.html", result=result, risk=risk)

@app.route("/raw_query", methods=["POST"])
def raw_query():
    raw_query = request.form["query"].strip()
    normalized = " ".join(raw_query.lower().split())

    # =========================
    # 1️⃣ SAFE EXECUTION CASE
    # =========================
    if normalized == "select * from users":
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users")
        rows = cur.fetchall()

        conn.close()

        return render_template(
            "results.html",
            rows=rows,
            table="users",
            risk=0
        )

    # =========================
    # 2️⃣ INTRUSION DETECTION ONLY
    # =========================

    # Table policy check (DO NOT assign risk here)
    allowed, tables = is_table_allowed(raw_query)

    # ML-based risk score
    X_input = prepare_input(raw_query)
    probability = model.predict_proba(X_input)[0][1]
    risk_score = round(probability * 100, 2)

    # =========================
    # 3️⃣ DECISION LOGIC
    # =========================

    # 🚨 High-risk query (true intrusion)
    if probability >= RISK_THRESHOLD:
        return render_template(
            "index.html",
            result="⚠ Query blocked due to high intrusion risk",
            risk=risk_score
        )

    # ⚠ Policy warning ONLY (not an attack)
    if not allowed:
        return render_template(
            "index.html",
            result="⚠ Query analyzed but execution not permitted by policy",
            risk=risk_score
        )

    # ✅ Normal low-risk query
    return render_template(
        "index.html",
        result="✅ Query analyzed successfully",
        risk=risk_score
    )

# =============================
# Run app
# =============================
if __name__ == "__main__":
    app.run(debug=True)
