"""
CSV Reading Service - Read CSV data from MongoDB GridFS.
"""

import csv
import io
from app.services.gridfs_service import download_file


def read_data_csv(file_path, icustayid):
    """
    Read CSV data from GridFS and filter by icustayid.
    
    Args:
        file_path: Original local path (will be converted to GridFS path)
        icustayid: ICU stay ID to filter by
    
    Returns:
        List of row dictionaries matching the icustayid
    """
    # Convert local path to GridFS path
    # e.g., 'app/./data/selected_data.csv' -> 'data/selected_data.csv'
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
    """
    Read entire CSV from GridFS as a pandas DataFrame.
    
    Args:
        file_path: Original local path or GridFS path
    
    Returns:
        pandas DataFrame
    """
    import pandas as pd
    
    # Convert local path to GridFS path
    gridfs_path = file_path.replace('app/', '').replace('./', '').lstrip('/')
    
    # Download CSV from GridFS
    csv_bytes = download_file(gridfs_path)
    
    return pd.read_csv(io.BytesIO(csv_bytes))