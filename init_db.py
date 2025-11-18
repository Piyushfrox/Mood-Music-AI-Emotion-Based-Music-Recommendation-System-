import sqlite3

conn = sqlite3.connect("users.db")
cur = conn.cursor()

# USERS TABLE
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password TEXT
)
""")

# EMOTIONS TABLE
cur.execute("""
CREATE TABLE IF NOT EXISTS emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    emotion TEXT,
    timestamp TEXT
)
""")

# PLAYLISTS TABLE
cur.execute("""
CREATE TABLE IF NOT EXISTS playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    emotion TEXT,
    link TEXT
)
""")

# Insert default admin user if not exists
cur.execute("SELECT * FROM users WHERE email='admin@example.com'")
if not cur.fetchone():
    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        ("admin", "admin@example.com", "admin")
    )

conn.commit()
conn.close()

print("✅ Database initialized successfully!")
print("➡ Default login: admin / admin")
