# app/routes/assets.py
from flask import Blueprint, Response, abort, current_app
from gridfs import GridFS
from pymongo import MongoClient
import certifi

assets_ = Blueprint('assets', __name__)

def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]  # Match config.py DBNAME
    return GridFS(db)

@assets_.route('/assets/images/<path:filename>')
def serve_image(filename):
    """Serve images from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    
    if not grid_file:
        abort(404)
    
    # Determine content type
    ext = filename.rsplit('.', 1)[-1].lower()
    content_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'webp': 'image/webp',
        'ico': 'image/x-icon',
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    return Response(
        grid_file.read(),
        content_type=content_type,
        headers={'Cache-Control': 'public, max-age=31536000'}
    )

@assets_.route('/assets/models/<path:filename>')
def serve_model(filename):
    """Serve model files from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    
    if not grid_file:
        abort(404)
    
    return Response(
        grid_file.read(),
        content_type='application/octet-stream'
    )

@assets_.route('/assets/data/<path:filename>')
def serve_data(filename):
    """Serve data files (CSV, JSON) from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    
    if not grid_file:
        abort(404)
    
    ext = filename.rsplit('.', 1)[-1].lower()
    content_types = {
        'csv': 'text/csv',
        'json': 'application/json',
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    return Response(
        grid_file.read(),
        content_type=content_type
    )
