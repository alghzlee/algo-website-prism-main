import random
import math


def generate_heart_rate(global_time):
    baseline = 75
    amplitude = 5
    frequency = 0.1
    noise = random.uniform(-0.25, 0.25)
    return max(min(baseline + amplitude * math.sin(frequency * global_time) + noise, 120), 50)
     

def generate_oxygen_saturation(global_time):
    amplitude = 2
    baseline = 98
    frequency = 0.2
    noise = random.uniform(-0.25, 0.25)
    return baseline + amplitude * math.sin(frequency * global_time + math.pi / 2) + noise
    

def generate_respiratory_rate(global_time):
    amplitude = 2
    baseline = 20
    frequency = 0.1
    noise = random.uniform(-0.5, 0.5)
    return baseline + amplitude * math.sin(frequency * global_time) + noise
    

# def generate_blood_pressure(global_time):
#     systolic_base = 120
#     diastolic_base = 80
#     fluctuation = math.sin(global_time * 0.1) * 10
#     return {
#         "systolic": systolic_base + fluctuation,
#         "diastolic": diastolic_base + fluctuation / 2
#     }

# def generate_temperature(global_time):
#     base_temp = 36.5
#     fluctuation = math.sin(global_time * 0.05) * 0.5
#     return round(base_temp + fluctuation, 1)

# def generate_renal():
#     creatinine = random.uniform(0.5, 6.0)
#     urine_output = random.randint(0, 1000)
#     if creatinine >= 5.0 or urine_output < 200:
#         return 4
#     elif creatinine >= 3.5 or urine_output < 500:
#         return 3
#     elif creatinine >= 2.0:
#         return 2
#     elif creatinine >= 1.2:
#         return 1
#     else:
#         return 0

# def generate_nervous():
#     gcs = random.randint(3, 15)
#     if gcs < 6:
#         return 4
#     elif gcs <= 9:
#         return 3
#     elif gcs <= 12:
#         return 2
#     elif gcs <= 14:
#         return 1
#     else:
#         return 0

# def generate_cardiovascular():
#     map_value = random.randint(50, 100)
#     vasopressors = random.choice(["none", "low", "medium", "high"])
#     if map_value < 70 and vasopressors == "high":
#         return 4
#     elif vasopressors == "medium":
#         return 3
#     elif vasopressors == "low":
#         return 2
#     elif map_value < 70:
#         return 1
#     else:
#         return 0

# def generate_liver():
#     bilirubin = random.uniform(0.5, 15.0)
#     if bilirubin >= 12.0:
#         return 4
#     elif bilirubin >= 6.0:
#         return 3
#     elif bilirubin >= 2.0:
#         return 2
#     elif bilirubin >= 1.2:
#         return 1
#     else:
#         return 0

# def generate_respiration():
#     pao2_fio2_ratio = random.randint(50, 500)
#     if pao2_fio2_ratio < 100:
#         return 4
#     elif pao2_fio2_ratio < 200:
#         return 3
#     elif pao2_fio2_ratio < 300:
#         return 2
#     elif pao2_fio2_ratio < 400:
#         return 1
#     else:
#         return 0

# def generate_coagulation():
#     platelet_count = random.randint(10, 200)
#     if platelet_count < 20:
#         return 4
#     elif platelet_count < 50:
#         return 3
#     elif platelet_count < 100:
#         return 2
#     elif platelet_count < 150:
#         return 1
#     else:
#         return 0

# def calculate_sofa_score():

#     score_renal = generate_renal()
#     score_nervous = generate_nervous()
#     score_cardiovascular = generate_cardiovascular()
#     score_liver = generate_liver()
#     score_respiration = generate_respiration()
#     score_coagulation = generate_coagulation()

#     total_score = (score_renal + score_nervous + score_cardiovascular +
#                    score_liver + score_respiration + score_coagulation)

#     return total_score
