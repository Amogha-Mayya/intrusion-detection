import sqlite3

conn = sqlite3.connect("database.db")
cur = conn.cursor()

# ------------------ USERS ------------------
users = [
    (1, "alice", "alice@gmail.com"),
    (2, "bob", "bob@gmail.com"),
    (3, "charlie", "charlie@gmail.com")
]

cur.executemany(
    "INSERT INTO users (id, username, email) VALUES (?, ?, ?)",
    users
)

# ------------------ PRODUCTS ------------------
products = [
    (1, "Laptop", 75000.50),
    (2, "Mouse", 500.75),
    (3, "Keyboard", 1200.00)
]

cur.executemany(
    "INSERT INTO products (id, name, price) VALUES (?, ?, ?)",
    products
)

# ------------------ ORDERS ------------------
orders = [
    (1, 1, 1),  # Alice ordered Laptop
    (2, 2, 2),  # Bob ordered Mouse
    (3, 3, 3)   # Charlie ordered Keyboard
]

cur.executemany(
    "INSERT INTO orders (id, user_id, product_id) VALUES (?, ?, ?)",
    orders
)

# ------------------ PAYMENTS ------------------
payments = [
    (1, 75000.50, "SUCCESS"),
    (2, 500.75, "SUCCESS"),
    (3, 1200.00, "FAILED")
]

cur.executemany(
    "INSERT INTO payments (id, amount, status) VALUES (?, ?, ?)",
    payments
)

# ------------------ LOGS ------------------
logs = [
    (1, "User alice created"),
    (2, "Order placed by bob"),
    (3, "Payment failed for charlie")
]

cur.executemany(
    "INSERT INTO logs (id, event) VALUES (?, ?)",
    logs
)

conn.commit()
conn.close()

print("Data inserted successfully!")
