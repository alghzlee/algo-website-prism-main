import csv

def read_data_csv(file_path, icustayid):
    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if float(row['icustayid']) == float(icustayid):
                data.append(row)
    return data