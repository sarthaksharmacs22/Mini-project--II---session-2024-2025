from flask import Flask, jsonify
import subprocess
from threading import Thread
import signal
import os
import time
import requests  # New import

app = Flask(__name__)

# Global variables
monitor_process = None
MONITOR_API_URL = "http://localhost:5001"  # Monitor's API port

def start_monitoring():
    global monitor_process
    if monitor_process is None:
        # Start monitor_engine.py which now includes its own API
        monitor_process = subprocess.Popen(
            ["python", "monitor_engine.py"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        print("Monitoring system started with PID:", monitor_process.pid)
        
        # Wait for monitor API to start
        time.sleep(2)  # Give it time to initialize

def stop_monitoring():
    global monitor_process
    if monitor_process:
        monitor_process.send_signal(signal.CTRL_BREAK_EVENT)
        monitor_process = None
        print("Monitoring system stopped")

@app.route('/')
def home():
    return """
    <h1>Exam Monitoring System</h1>
    <p>Endpoints:</p>
    <ul>
        <li><a href="/start">/start</a> - Start monitoring</li>
        <li><a href="/stop">/stop</a> - Stop monitoring</li>
        <li><a href="/activities">/activities</a> - View detections</li>
        <li><a href="/status">/status</a> - Check status</li>
    </ul>
    """

@app.route('/start')
def start():
    Thread(target=start_monitoring).start()
    return jsonify({"status": "Monitoring started"})

@app.route('/stop')
def stop():
    stop_monitoring()
    return jsonify({"status": "Monitoring stopped"})

@app.route('/status')
def status():
    try:
        # Try to contact the monitor's API
        response = requests.get(f"{MONITOR_API_URL}/get_activities", timeout=2)
        return jsonify({
            "running": True,
            "detections_count": len(response.json().get('activities', [])),
            "last_event": response.json().get('activities', [])[-1] if response.json().get('activities') else None
        })
    except:
        return jsonify({"running": False})

@app.route('/activities')
def activities():
    try:
        response = requests.get(f"{MONITOR_API_URL}/get_activities", timeout=3)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({"error": "Monitor API unavailable"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Could not connect to monitoring system",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)