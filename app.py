import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")  # For Flask
import matplotlib.pyplot as plt

# OUR EMOTION MODEL
from emotion_detector import detect_emotion


app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------
def get_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
@app.route("/")
def home():
    if "username" not in session:
        return redirect("/login")
    return render_template("index.html", username=session["username"])


# ---------------------------------------------------
# SIGNUP PAGE
# ---------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            return render_template("signup.html", error="Email already exists!")

        cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, password))
        conn.commit()

        return redirect("/login")

    return render_template("signup.html")


# ---------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        row = cur.fetchone()

        if row:
            session["username"] = row["username"]
            session["email"] = row["email"]
            return redirect("/")
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


# ---------------------------------------------------
# LOGOUT
# ---------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------------------------------------------
# EMOTION PREDICTION
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files["image"]
        if not img_file:
            return jsonify({"error": "No image received"}), 400

        filename = secure_filename(img_file.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img_file.save(img_path)

        # --- GET RAW EMOTION FROM MODEL ---
        raw_emo = detect_emotion(img_path)

        # --- CLEAN EMOTION ---
        if isinstance(raw_emo, dict):
            emotion = raw_emo.get("emotion", "NEUTRAL")
        elif isinstance(raw_emo, (list, tuple)):
            emotion = str(raw_emo[0])
        else:
            emotion = str(raw_emo)

        # --- SAVE EMOTION TO DATABASE ---
        email = session.get("email")
        if email:
            conn = get_db()
            conn.execute(
                "INSERT INTO emotions (email, emotion, timestamp) VALUES (?, ?, ?)",
                (email, emotion, datetime.now())
            )
            conn.commit()

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------
# MOOD ANALYTICS (FULL WORKING CHART SYSTEM)
# ---------------------------------------------------
@app.route("/emotion_stats")
def emotion_stats():
    if "email" not in session:
        return redirect("/login")

    email = session["email"]
    days = int(request.args.get("range", 7))

    # Fetch mood data
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT emotion, timestamp FROM emotions WHERE email=? ORDER BY timestamp",
        (email,)
    ).fetchall()

    if not rows:
        return render_template("emotion_stats.html", chart_data=None, pie_data=None, days=days)

    timestamps = [row["timestamp"] for row in rows]
    emotions = [row["emotion"] for row in rows]

    # Convert emotions to numeric scale
    emotion_levels = {"SAD": 0, "NEUTRAL": 1, "HAPPY": 2}
    levels = [emotion_levels[e] for e in emotions]

    # ---------------------------------------------------
    # LINE CHART
    # ---------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, levels, marker="o", linewidth=3, color="#CCFF00")

    plt.yticks([0, 1, 2], ["SAD", "NEUTRAL", "HAPPY"], fontsize=14, color="#CCFF00")
    plt.xticks(rotation=45, fontsize=10, color="#CCFF00")
    plt.grid(True, alpha=0.3)
    plt.title("Emotions Over Time", fontsize=16, color="#CCFF00")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor="#000000")
    buf.seek(0)
    chart_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------
    plt.figure(figsize=(6, 6))
    emotion_counts = {e: emotions.count(e) for e in set(emotions)}

    plt.pie(
        emotion_counts.values(),
        labels=emotion_counts.keys(),
        autopct='%1.1f%%',
        colors=["#CCFF00", "#FF007F", "#00FFFF"],
        textprops={'color': 'white'}
    )
    plt.title("Emotion Distribution", color="#CCFF00")

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor="#000000")
    buf.seek(0)
    pie_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return render_template("emotion_stats.html", chart_data=chart_data, pie_data=pie_data, days=days)


# ---------------------------------------------------
# LIKE PLAYLIST
# ---------------------------------------------------
@app.route("/like_song", methods=["POST"])
def like_song():
    emotion = request.form["emotion"]
    link = request.form["song_link"]
    email = session.get("email")

    if not email:
        return jsonify({"message": "Login required"}), 403

    conn = get_db()
    conn.execute("INSERT INTO playlists (email, emotion, link) VALUES (?, ?, ?)",
                 (email, emotion, link))
    conn.commit()

    return jsonify({"message": "Playlist saved!"})


# ---------------------------------------------------
# VIEW SAVED PLAYLISTS
# ---------------------------------------------------
@app.route("/my_playlists")
def my_playlists():
    email = session.get("email")
    if not email:
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT emotion, link FROM playlists WHERE email=?", (email,))
    data = cur.fetchall()

    playlists = [(row["emotion"], row["link"]) for row in data]
    return render_template("my_playlists.html", playlists=playlists)


# ---------------------------------------------------
# REMOVE SAVED PLAYLIST
# ---------------------------------------------------
@app.route("/remove_playlist", methods=["POST"])
def remove_playlist():
    link = request.form["song_link"]

    conn = get_db()
    conn.execute("DELETE FROM playlists WHERE link=?", (link,))
    conn.commit()

    return jsonify({"message": "Playlist removed!"})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
