from pymongo import MongoClient
from gridfs import GridFS
from flask import current_app
import os
import certifi

def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    # Gunakan certifi untuk SSL certificate (required untuk MongoDB Atlas)
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DB_NAME']]
    return GridFS(db)

def upload_file(file_path, filename=None):
    """Upload file to MongoDB Atlas GridFS"""
    fs = get_gridfs()
    if filename is None:
        filename = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        file_id = fs.put(f, filename=filename)
    
    return file_id

def get_file(filename):
    """Get file from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    return fs.find_one({'filename': filename})

def download_file(filename):
    """Download file content from GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    if grid_file:
        return grid_file.read()
    return None