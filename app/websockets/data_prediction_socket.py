from flask import Blueprint
from flask_socketio import emit
from app.services.data_prediction_service import generate_heart_rate, generate_oxygen_saturation, generate_respiratory_rate
from app.extensions import socketio
import time

dataPredictionSocketio = Blueprint('data_prediction', __name__)
global_time = 0

@socketio.on('connect', namespace='/vital_patient_prediction')
def handle_connect():
    emit('message', {'status': 'Connected to data vital prediction'})

@socketio.on('get_data_prediction_vital_patient', namespace='/vital_patient_prediction')
def handle_get_data():
    global global_time
    while True:
        heart_rate = generate_heart_rate(global_time)
        oxygen_saturation = generate_oxygen_saturation(global_time)
        respiratory_rate = generate_respiratory_rate(global_time)

        data = {
            "heart_rate": heart_rate,
            "oxygen_saturation": oxygen_saturation,
            "respiratory_rate": respiratory_rate,
        }
        emit('data_prediction_vital_patient', data, namespace='/vital_patient_prediction')

        global_time += 1
        time.sleep(5)  
