import csv
import io
from app.services.gridfs_service import download_file


def read_data_csv(file_path, icustayid):
    gridfs_path = file_path.replace('app/', '').replace('./', '').lstrip('/')
    
    # Download CSV from GridFS
    csv_bytes = download_file(gridfs_path)
    csv_content = csv_bytes.decode('utf-8')
    
    data = []
    csv_reader = csv.DictReader(io.StringIO(csv_content))
    for row in csv_reader:
        if float(row['icustayid']) == float(icustayid):
            data.append(row)
    return data


def read_csv_as_dataframe(file_path):
    import pandas as pd
    
    gridfs_path = file_path.replace('app/', '').replace('./', '').lstrip('/')
    
    csv_bytes = download_file(gridfs_path)
    
    return pd.read_csv(io.BytesIO(csv_bytes))