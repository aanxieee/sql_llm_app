import sqlite3

# data.db file isi folder mein ban jayegi
conn = sqlite3.connect("data.db")
cur = conn.cursor()

# Example table: sales
cur.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT,
    product TEXT,
    quantity INTEGER,
    price_per_unit REAL,
    sale_date TEXT
);
""")

# Kuch sample data
rows = [
    ("Aanya", "Notebook", 3, 50.0, "2025-11-15"),
    ("Aarna", "Pen", 10, 10.0, "2025-11-16"),
    ("Rahul", "Notebook", 1, 50.0, "2025-11-16"),
    ("Aanya", "Pencil", 5, 5.0, "2025-11-17"),
]

cur.executemany(
    "INSERT INTO sales (customer_name, product, quantity, price_per_unit, sale_date) VALUES (?, ?, ?, ?, ?)",
    rows,
)

conn.commit()
conn.close()

print("data.db created with 'sales' table and sample data ✅")
