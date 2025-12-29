from app.extensions import socketio
from flask import Blueprint
from flask_socketio import emit
from app.services.read_csv_services import read_data_csv

treatmentRecommendationSocketio = Blueprint('treatment_recommendation', __name__)

file_path_dataset_monitoring_patient = 'app/data/df_with_readable_charttime.csv'

data_monitoring_patient = {}

@socketio.on('connect', namespace='/monitoring_patient')
def handle_connect():
    emit('message', {'status': 'Connected to data vital patient'})

@socketio.on('get_data_monitoring_patient', namespace='/monitoring_patient')
def handle_get_data(data):
    global data_monitoring_patient
    icustayid = data.get('icustayid')

    if icustayid not in data_monitoring_patient:
        data_monitoring_patient[icustayid] = 0

    vital_data = read_data_csv(file_path_dataset_monitoring_patient, icustayid)

    while True:
        index = data_monitoring_patient[icustayid]

        if len(vital_data) == 0:
            emit('data_monitoring_patient', {'error': 'No data available for this patient'})
            break

        row = vital_data[index]

        emit('data_monitoring_patient', row)

        socketio.sleep(5) 

        data_monitoring_patient[icustayid] = (index + 1) % len(vital_data)