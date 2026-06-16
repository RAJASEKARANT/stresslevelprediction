import csv
import random
import os

print(f"Current CWD: {os.getcwd()}")
output_path = os.path.join(os.getcwd(), 'stress.csv')
print(f"Writing to: {output_path}")

# Set random seed
random.seed(42)

# Generate synthetic data
n_samples = 200

headers = ['heart_rate', 'sleeping_hours', 'sleep_quality', 'blood_oxygen', 'body_temperature', 'physical_activity', 'stress_level']

rows = []

for _ in range(n_samples):
    # Heart Rate: Normal 60-100, High > 100
    heart_rate = random.randint(60, 120)

    # Sleeping Hours: 4-10 hours
    sleeping_hours = random.uniform(4, 10)

    # Sleep Quality: 1-10 scale
    sleep_quality = random.randint(1, 10)

    # Blood Oxygen (SpO2): 90-100%
    blood_oxygen = random.uniform(90, 100)

    # Body Temperature: 36.0 - 37.5 Celsius
    body_temperature = random.uniform(36.0, 37.5)

    # Physical Activity: 0-100
    physical_activity = random.randint(0, 100)

    # Stress Score Calculation
    # Normalize inputs roughly to 0-1 for scoring
    # HR: (val - 60) / 60
    s_hr = (heart_rate - 60) / 60

    # Sleep: (10 - val) / 6
    s_sleep = (10 - sleeping_hours) / 6

    # Quality: (10 - val) / 9
    s_qual = (10 - sleep_quality) / 9

    # SpO2: (100 - val) / 10
    s_oxy = (100 - blood_oxygen) / 10

    # Activity: (100 - val) / 100
    s_act = (100 - physical_activity) / 100

    # Weighted sum
    raw_score = (s_hr * 0.3) + (s_sleep * 0.3) + (s_qual * 0.2) + (s_oxy * 0.1) + (s_act * 0.1)

    # Add noise
    raw_score += random.gauss(0, 0.05)

    # Determine level
    if raw_score < 0.33:
        stress_level = 0 # Low
    elif raw_score < 0.66:
        stress_level = 1 # Medium
    else:
        stress_level = 2 # High

    rows.append([
        heart_rate,
        round(sleeping_hours, 2),
        sleep_quality,
        round(blood_oxygen, 2),
        round(body_temperature, 2),
        physical_activity,
        stress_level
    ])

# Write to CSV
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print("stress.csv generated successfully.")
