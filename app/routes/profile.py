from flask import request, render_template, current_app, Blueprint, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import jwt
from app.middleware.authenticate import token_required, admin_required
from app.services.gridfs_service import upload_file, delete_file, file_exists

profile_ = Blueprint('profile', __name__)


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
            if file.filename:
                filename = secure_filename(file.filename)
                extension = filename.split(".")[-1]
                
                # Generate unique filename for GridFS
                gridfs_filename = f"profiles/{email.replace('@', '_at_').replace('.', '_')}.{extension}"
                
                # Delete existing profile picture if exists
                if file_exists(gridfs_filename):
                    delete_file(gridfs_filename)
                
                # Upload to GridFS
                file_id = upload_file(file, gridfs_filename)
                
                newDoc["profile"] = filename
                newDoc["profilePict"] = gridfs_filename
            
        current_app.db.users.update_one(
            {"email": payload["id"]}, {"$set": newDoc})
        return jsonify({"msg": "Profile successfully updated!"})
    except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
        return redirect(url_for("profile"))


@profile_.route('/user-management')
@admin_required
def user_management():
    all_users = list(current_app.db.users.find({}))
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        logged_in_user = current_app.db.users.find_one(
            {"email": payload["id"]})
        if logged_in_user:
            return render_template('profile/user-management.html',
                                   users=all_users,
                                   user_info=logged_in_user)
        else:
            return "User not found", 404
    except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
        return redirect(url_for("auth.sign_in", msg="Please login first!"))
