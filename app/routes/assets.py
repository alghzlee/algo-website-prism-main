"""
Assets Routes - Serve files from MongoDB GridFS and Admin Panel for file management.
"""

from flask import Blueprint, Response, request, render_template, jsonify, current_app, redirect, url_for
import jwt
from app.services.gridfs_service import (
    get_gridfs, upload_file, delete_file, download_file, 
    list_files, file_exists, get_file
)
import mimetypes

assets_ = Blueprint('assets', __name__)


# ============================================================================
# PUBLIC ROUTES - Serve files from GridFS
# ============================================================================

@assets_.route('/assets/<path:filepath>')
def serve_asset(filepath):
    """
    Serve any file from GridFS.
    Example: /assets/images/assets/bed-hospital.jpg
             /assets/data/selected_data.csv
             /assets/profiles/user_email.jpg
    """
    try:
        file_data = get_file(filepath)
        if file_data is None:
            return Response("File not found", status=404)
        
        content = file_data.read()
        content_type = file_data.content_type or 'application/octet-stream'
        
        # Disable cache for profile images so updates show immediately
        if filepath.startswith('profiles/'):
            cache_control = 'no-cache, no-store, must-revalidate'
        else:
            cache_control = 'public, max-age=31536000'  # Cache for 1 year
        
        return Response(
            content,
            mimetype=content_type,
            headers={
                'Cache-Control': cache_control
            }
        )
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


# ============================================================================
# ADMIN ROUTES - File Management Panel
# ============================================================================

def admin_required(f):
    """Decorator to require admin authentication."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        myToken = request.cookies.get("mytoken")
        SECRET_KEY = current_app.config['SECRET_KEY']
        try:
            payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
            user_info = current_app.db.users.find_one({"email": payload["id"]})
            if not user_info:
                return redirect(url_for("auth.sign_in", msg="Please login first!"))
            # Add user_info to request context
            request.user_info = user_info
            return f(*args, **kwargs)
        except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
            return redirect(url_for("auth.sign_in", msg="Please login first!"))
    return decorated_function


@assets_.route('/admin/assets')
@admin_required
def admin_assets_page():
    """Render admin assets management page."""
    user_info = getattr(request, 'user_info', None)
    return render_template('admin/admin-assets.html', user_info=user_info)


@assets_.route('/admin/assets/list', methods=['GET'])
@admin_required
def admin_list_files():
    """List all files in GridFS."""
    try:
        prefix = request.args.get('prefix', None)
        files = list_files(prefix)
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@assets_.route('/admin/assets/upload', methods=['POST'])
@admin_required
def admin_upload_file():
    """Upload file(s) to GridFS."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        files = request.files.getlist('file')
        folder = request.form.get('folder', '')
        
        uploaded = []
        for file in files:
            if file.filename:
                # Build filename with folder path
                if folder:
                    filename = f"{folder.strip('/')}/{file.filename}"
                else:
                    filename = file.filename
                
                # Upload to GridFS
                file_id = upload_file(file, filename)
                uploaded.append({
                    'filename': filename,
                    'id': str(file_id)
                })
        
        return jsonify({
            'success': True,
            'uploaded': uploaded,
            'count': len(uploaded)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@assets_.route('/admin/assets/delete', methods=['DELETE'])
@admin_required
def admin_delete_file():
    """Delete file from GridFS."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400
        
        deleted = delete_file(filename)
        
        if deleted:
            return jsonify({'success': True, 'message': f'Deleted: {filename}'})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@assets_.route('/admin/assets/check', methods=['POST'])
@admin_required
def admin_check_file():
    """Check if file exists in GridFS."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400
        
        exists = file_exists(filename)
        
        return jsonify({
            'success': True,
            'exists': exists,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
