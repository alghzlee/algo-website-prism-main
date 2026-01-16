from flask import Blueprint, current_app
from flask_socketio import emit
from app.extensions import socketio
from app.services.read_data_mongo_services import read_data_mongo

treatmentRecommendationSocketio = Blueprint(
    'treatment_recommendation', __name__
)

# menyimpan index streaming per icustayid
data_monitoring_patient = {}


def get_monitoring_collection():
    """Get MongoDB collection lazily at runtime using Flask app context."""
    return current_app.db["df_with_readable_charttime"]


@socketio.on('connect', namespace='/monitoring_patient')
def handle_connect():
    emit('message', {'status': 'Connected to data vital patient'})


@socketio.on('get_data_monitoring_patient', namespace='/monitoring_patient')
def handle_get_data(data):
    global data_monitoring_patient

    icustayid = data.get('icustayid')

    if icustayid is None:
        emit(
            'data_monitoring_patient',
            {'error': 'icustayid is required'}
        )
        return

    if icustayid not in data_monitoring_patient:
        data_monitoring_patient[icustayid] = 0

    # Ambil data dari MongoDB (sekali saja)
    vital_data = read_data_mongo(get_monitoring_collection(), icustayid)

    if len(vital_data) == 0:
        emit(
            'data_monitoring_patient',
            {'error': 'No data available for this patient'}
        )
        return

    while True:
        index = data_monitoring_patient[icustayid]

        row = vital_data[index]
        emit('data_monitoring_patient', row)

        socketio.sleep(5)

        data_monitoring_patient[icustayid] = (index + 1) % len(vital_data)
