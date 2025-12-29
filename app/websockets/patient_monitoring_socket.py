from app.extensions import socketio
from flask import Blueprint
from flask_socketio import emit
from app.services.monitoring_patient_service import read_detail_patient, sofa_respiratory, sofa_coagulation, sofa_liver, sofa_cardiovascular, sofa_neurological, sofa_renal, calculate_sofa_score
from app.services.read_csv_services import read_data_csv

patientMonitoringSocketio = Blueprint('patient_monitoring', __name__)

file_path_dataset_vital_patient = 'app/data/selected_data.csv'
file_path_dataset_sofal_patient = 'app/data/sofa_indicators.csv'

data_vital_patient = {}
data_sofa_patient = {}

@socketio.on('connect', namespace='/detail_patient')
def handle_connect():
    emit('message', {'status': 'Connected to patient detail'})


@socketio.on('get_detail_patient', namespace='/detail_patient')
def handle_get_data(data):
    icustayid = data.get('icustayid')
    detail_patient = read_detail_patient(float(icustayid))
    emit('detail_patient_data', detail_patient)
    
@socketio.on('connect', namespace='/vital_patient')
def handle_connect():
    emit('message', {'status': 'Connected to data vital patient'})

@socketio.on('get_data_vital_patient', namespace='/vital_patient')
def handle_get_data(data):
    global data_vital_patient
    icustayid = data.get('icustayid')

    if icustayid not in data_vital_patient:
        data_vital_patient[icustayid] = 0

    vital_data = read_data_csv(file_path_dataset_vital_patient, icustayid)

    while True:
        index = data_vital_patient[icustayid]

        if len(vital_data) == 0:
            emit('data_vital_patient', {'error': 'No data available for this patient'})
            break

        row = vital_data[index]

        emit('data_vital_patient', row)

        socketio.sleep(5) 

        data_vital_patient[icustayid] = (index + 1) % len(vital_data)
        
@socketio.on('connect', namespace='/sofa_patient')
def handle_connect():
    emit('message', {'status': 'Connected to data sofa score patient'})

@socketio.on('get_data_sofa_patient', namespace='/sofa_patient')
def handle_get_data(data):
    global data_sofa_patient
    icustayid = data.get('icustayid')

    if icustayid not in data_sofa_patient:
        data_sofa_patient[icustayid] = 0

    sofa_data = read_data_csv(file_path_dataset_sofal_patient, icustayid)

    while True:
        index = data_sofa_patient[icustayid]

        if len(sofa_data) == 0:
            emit('data_sofa_patient', {'error': 'No data available for this patient'})
            break

        row = sofa_data[index]
        
        respiratory = sofa_respiratory(float(row['Respiratory']))
        coagulation = sofa_coagulation(float(row['Coagulation']))
        liver = sofa_liver(float(row['Liver']))
        cardiovascular = sofa_cardiovascular(float(row['Cardiovascular']))
        neurological = sofa_neurological(float(row['Neurological']))
        renal = sofa_renal(float(row['Renal']))
        sofa_score = calculate_sofa_score(respiratory, coagulation, liver, cardiovascular, neurological, renal)

        update_sofa_data = {
            'sofa_score': sofa_score,
            'respiratory': respiratory,
            'coagulation': coagulation,
            'liver': liver,
            'cardiovascular': cardiovascular,
            'neurological': neurological,
            'renal': renal
        }
    
        emit('data_sofa_patient', update_sofa_data)

        if sofa_score > 6:
            emit('notification', {'message': 'Sofa Score melebihi batas!', 'sofa_score': sofa_score})
            
        socketio.sleep(5) 

        data_sofa_patient[icustayid] = (index + 1) % len(sofa_data)