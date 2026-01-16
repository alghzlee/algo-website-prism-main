import random
import math


def generate_heart_rate(global_time):
    """Generate simulated heart rate data based on time."""
    baseline = 75
    amplitude = 5
    frequency = 0.1
    noise = random.uniform(-0.25, 0.25)
    return max(min(baseline + amplitude * math.sin(frequency * global_time) + noise, 120), 50)
     

def generate_oxygen_saturation(global_time):
    """Generate simulated oxygen saturation data based on time."""
    amplitude = 2
    baseline = 98
    frequency = 0.2
    noise = random.uniform(-0.25, 0.25)
    return baseline + amplitude * math.sin(frequency * global_time + math.pi / 2) + noise
    

def generate_respiratory_rate(global_time):
    """Generate simulated respiratory rate data based on time."""
    amplitude = 2
    baseline = 20
    frequency = 0.1
    noise = random.uniform(-0.5, 0.5)
    return baseline + amplitude * math.sin(frequency * global_time) + noise
