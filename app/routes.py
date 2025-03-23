from flask import render_template
from app import app, socketio
from flask_socketio import emit

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('sensor_data')
def handle_sensor_data(data):
    emit('update_data', data, broadcast=True)