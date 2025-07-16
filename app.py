from flask import Flask, render_template, jsonify, make_response
from livereload import Server
import threading
import time
import datetime
import subprocess
import sys

from model_predictions import predict_row
from api_gemini import get_power_insight_response  # <- Your Gemini logic here

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

latest_data = {}
last_updated = None
power_insight = "Loading power insight..."  # Default placeholder


# === Background thread: update prediction every 5 minutes ===
def update_prediction_periodically():
    global latest_data, last_updated
    while True:
        latest_data = predict_row()
        last_updated = datetime.datetime.now().isoformat()
        print("Prediction updated at", last_updated)
        time.sleep(300)


# === Background thread: update Gemini insight every hour ===
def update_insight_hourly():
    global power_insight
    while True:
        print("Getting new power insight from Gemini...")
        try:
            power_insight = get_power_insight_response(
                predicted_power=latest_data.get("predicted_total_power", 362.27),
                current_hour=datetime.datetime.now().hour,
                current_minute=datetime.datetime.now().minute
            )
        except Exception as e:
            print("Failed to get insight:", e)
        time.sleep(3000)


# === Background thread: update graph ===
def update_image_periodically():
    while True:
        subprocess.run([sys.executable, "new_grapg_for_Energy_Overview.py"])
        time.sleep(300)


# === Main route ===
@app.route("/")
def home():
    response = make_response(render_template("index.html", power_insight=power_insight, **latest_data))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    threading.Thread(target=update_prediction_periodically, daemon=True).start()
    threading.Thread(target=update_image_periodically, daemon=True).start()
    threading.Thread(target=update_insight_hourly, daemon=True).start()  # <- NEW INSIGHT THREAD

    server = Server(app.wsgi_app)
    server.watch('templates/')
    server.watch('static/')
    server.serve(port=9000, debug=True)
