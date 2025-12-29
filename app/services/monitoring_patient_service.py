patients = {
    50100001: {"name": "Abdul Kirom", "age": "60 years old", "body_period": "169 cm 73 kg"},
    50100002: {"name": "Dewi Rahmawati", "age": "45 years old", "body_period": "172 cm 68 kg"},
    50100003: {"name": "Rizal Abdi", "age": "30 years old", "body_period": "160 cm 55 kg"},
    50100004: {"name": "Lia Rahma", "age": "39 years old", "body_period": "180 cm 80 kg"},
    50100005: {"name": "Prasetyo Putra", "age": "55 years old", "body_period": "176 cm 65 kg"},
    50100006: {"name": "Aldo Akfar Rahma", "age": "49 years old", "body_period": "174 cm 73 kg"},
    50100007: {"name": "Kusmita Suryah", "age": "48 years old", "body_period": "166 cm 63 kg"},
    50100008: {"name": "Aldy Hamzah", "age": "68 years old", "body_period": "175 cm 80 kg"},
}

def read_detail_patient(icustayid):
    patient_detail = patients.get(icustayid)
    if patient_detail:
        return patient_detail
    else:
        return {"error": "Patient not found"}


def sofa_respiratory(pao2_fio2_ratio, ventilasi_mekanis=False):
    if pao2_fio2_ratio >= 400:
        return 0
    elif pao2_fio2_ratio < 400 and pao2_fio2_ratio >= 300:
        return 1
    elif pao2_fio2_ratio < 300 and pao2_fio2_ratio >= 200:
        return 2
    elif pao2_fio2_ratio < 200 and ventilasi_mekanis:
        return 3
    elif pao2_fio2_ratio < 100 and ventilasi_mekanis:
        return 4
    else:
        return 0

def sofa_coagulation(platelet_count):
    if platelet_count >= 150:
        return 0
    elif platelet_count < 150 and platelet_count >= 100:
        return 1
    elif platelet_count < 100 and platelet_count >= 50:
        return 2
    elif platelet_count < 50 and platelet_count >= 20:
        return 3
    elif platelet_count < 20:
        return 4
    else:
        return 0

def sofa_liver(bilirubin):
    if bilirubin < 1.2:
        return 0
    elif 1.2 <= bilirubin < 2.0:
        return 1
    elif 2.0 <= bilirubin < 6.0:
        return 2
    elif 6.0 <= bilirubin < 12.0:
        return 3
    elif bilirubin >= 12.0:
        return 4
    else:
        return 0

def sofa_cardiovascular(map_value, vasopressor=None):
    if map_value >= 70:
        return 0
    elif map_value < 70:
        return 1
    elif vasopressor == 'dopamine_low' or vasopressor == 'dobutamine':
        return 2
    elif vasopressor == 'dopamine_medium' or vasopressor == 'norepinephrine_low':
        return 3
    elif vasopressor == 'dopamine_high' or vasopressor == 'norepinephrine_high':
        return 4
    else:
        return 0


def sofa_neurological(gcs):
    if gcs == 15:
        return 0
    elif 13 <= gcs < 15:
        return 1
    elif 10 <= gcs < 13:
        return 2
    elif 6 <= gcs < 10:
        return 3
    elif gcs < 6:
        return 4
    else:
        return 0

def sofa_renal(creatinine, urine_output=None):
    if creatinine < 1.2:
        return 0
    elif 1.2 <= creatinine < 2.0:
        return 1
    elif 2.0 <= creatinine < 3.5:
        return 2
    elif 3.5 <= creatinine < 5.0 or (urine_output is not None and urine_output < 500):
        return 3
    elif creatinine >= 5.0 or (urine_output is not None and urine_output < 200):
        return 4
    else:
        return 0


def calculate_sofa_score(sofa_respiratory, sofa_coagulation, sofa_liver, sofa_cardiovascular, sofa_neurological, sofa_renal):
    sofa_score = sofa_respiratory + sofa_coagulation + sofa_liver + sofa_cardiovascular + sofa_neurological + sofa_renal
    return sofa_score