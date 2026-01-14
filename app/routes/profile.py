from flask import Flask, request, render_template, current_app, Blueprint, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import jwt
from app.middleware.authenticate import token_required
from gridfs import GridFS
from pymongo import MongoClient
import certifi

profile_ = Blueprint('profile', __name__)


def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]
    return GridFS(db)


@profile_.route('/profile')
def profile():
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        user_info = current_app.db.users.find_one({"email": payload["id"]})
        return render_template('profile/profile.html', user_info=user_info)
    except jwt.ExpiredSignatureError:
        return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
    except jwt.exceptions.DecodeError:
        return redirect(url_for("auth.sign_in", msg="Please login first!"))


@profile_.route('/update-profile', methods=["POST"])
@token_required
def update_profile():
    SECRET_KEY = current_app.config['SECRET_KEY']
    token_receive = request.cookies.get("mytoken")
    try:
        payload = jwt.decode(token_receive, SECRET_KEY, algorithms=["HS256"])
        username = request.form["username"]
        email = payload["id"]
        newDoc = {"username": username}
        
        if "filePict" in request.files:
            file = request.files["filePict"]
            if file.filename:  # Check if file was actually uploaded
                filename = secure_filename(file.filename)
                extension = filename.split(".")[-1].lower()
                
                # Generate unique filename for GridFS
                gridfs_filename = f"profile_{email.replace('@', '_at_').replace('.', '_')}.{extension}"
                
                # Upload to MongoDB GridFS
                fs = get_gridfs()
                
                # Delete existing profile picture if exists
                existing = fs.find_one({'filename': gridfs_filename})
                if existing:
                    fs.delete(existing._id)
                
                # Also try to delete with old naming convention
                for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    old_filename = f"profile_{email.replace('@', '_at_').replace('.', '_')}.{ext}"
                    old_file = fs.find_one({'filename': old_filename})
                    if old_file:
                        fs.delete(old_file._id)
                
                # Upload new file
                fs.put(file, filename=gridfs_filename)
                
                newDoc["profile"] = filename
                newDoc["profilePict"] = gridfs_filename  # Store GridFS filename
        
        current_app.db.users.update_one(
            {"email": payload["id"]}, {"$set": newDoc})
        return jsonify({"msg": "Profile successfully updated!"})
    except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
        return redirect(url_for("profile"))


@profile_.route('/user-settings')
@token_required
def user_settings():
    all_users = list(current_app.db.users.find({}))
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        logged_in_user = current_app.db.users.find_one(
            {"email": payload["id"]})
        if logged_in_user:
            return render_template('profile/user-settings.html',
                                   users=all_users,
                                   user_info=logged_in_user)
        else:
            return "User not found", 404
    except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
        return redirect(url_for("auth.sign_in", msg="Please login first!"))
