"""
GridFS Service - Centralized service for all MongoDB GridFS operations.
Handles upload, download, delete, and listing of files stored in GridFS.
"""

import io
import tempfile
import mimetypes
from flask import current_app
from pymongo import MongoClient
from gridfs import GridFS
import certifi


def get_gridfs():
    """
    Get GridFS instance from MongoDB Atlas with SSL support.
    Uses the app's database configuration.
    """
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]
    return GridFS(db)


def upload_file(file_data, filename, content_type=None):
    fs = get_gridfs()
    
    # Delete existing file with same name if exists
    existing = fs.find_one({'filename': filename})
    if existing:
        fs.delete(existing._id)
    
    # Determine content type if not provided
    if content_type is None:
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = 'application/octet-stream'
    
    # Handle both file objects and bytes
    if hasattr(file_data, 'read'):
        data = file_data.read()
        file_data.seek(0)  # Reset file pointer
    else:
        data = file_data
    
    return fs.put(data, filename=filename, content_type=content_type)


def delete_file(filename):
    fs = get_gridfs()
    file = fs.find_one({'filename': filename})
    if file:
        fs.delete(file._id)
        return True
    return False


def get_file(filename):
    fs = get_gridfs()
    return fs.find_one({'filename': filename})


def download_file(filename):
    fs = get_gridfs()
    file = fs.find_one({'filename': filename})
    if file is None:
        raise FileNotFoundError(f"File not found in GridFS: {filename}")
    return file.read()


def download_to_tempfile(filename, suffix=None):
    if suffix is None:
        # Extract suffix from filename
        if '.' in filename:
            suffix = '.' + filename.split('.')[-1]
    
    content = download_file(filename)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(content)
    temp_file.close()
    
    return temp_file.name


def list_files(prefix=None):
    fs = get_gridfs()
    files = []
    
    query = {}
    if prefix:
        query['filename'] = {'$regex': f'^{prefix}'}
    
    for file in fs.find(query):
        files.append({
            'filename': file.filename,
            'size': file.length,
            'content_type': file.content_type,
            'upload_date': file.upload_date.isoformat() if file.upload_date else None,
            '_id': str(file._id)
        })
    
    return files


def file_exists(filename):
    fs = get_gridfs()
    return fs.find_one({'filename': filename}) is not None
