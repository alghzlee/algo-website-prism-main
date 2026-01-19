from datetime import datetime, timedelta
from flask import request, render_template, current_app, Blueprint, jsonify, redirect, url_for, make_response
import hashlib
import jwt
from app.middleware.authenticate import token_required, admin_required
from bson import ObjectId


auth_ = Blueprint('auth', __name__)


@auth_.route('/sign-in')
def sign_in():
    msg = request.args.get("msg")
    return render_template('auth/login.html', msg=msg)


@auth_.route("/sign-in/check", methods=["POST"])
def sign_in_check():
    email = request.form["email"]
    password = request.form["password"]
    pw_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    SECRET_KEY = current_app.config['SECRET_KEY']
    result = current_app.db.users.find_one(
        {
            "email": email,
            "password": pw_hash,
        }
    )
    if result:
        payload = {
            "id": email,
            "exp": datetime.utcnow() + timedelta(hours=8)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify(
            {
                "result": "success",
                "token": token,
            }
        )
    else:
        return jsonify(
            {
                "result": "fail",
                "msg": "Wrong email or password!",
            }
        )


@auth_.route('/sign-up')
def sign_up():
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        user_info = current_app.db.users.find_one({"email": payload["id"]})
        return render_template('auth/register.html', user_info=user_info)
    except jwt.ExpiredSignatureError:
        return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
    except jwt.exceptions.DecodeError:
        return redirect(url_for("auth.sign_in", msg="Please login first!"))


@auth_.route("/sign-up/save", methods=["POST"])
@token_required
def sign_up_save():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    role = request.form.get('role')
    address = request.form.get('address', '')
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Generate default avatar using UI Avatars API
    default_avatar = f"https://ui-avatars.com/api/?name={username}&background=0891b2&color=fff&size=128"
    
    doc = {
        "username": username,
        "email": email,
        "password": password_hash,
        "role": role,
        "address": address,
        "profile": "",
        "profilePict": default_avatar
    }
    exists = bool(current_app.db.users.find_one({"email": email}))
    if exists == False:
        current_app.db.users.insert_one(doc)

    return jsonify({'exists': exists})


@auth_.route('/users/delete/email/<email>', methods=['DELETE'])
@admin_required
def delete_user_by_email(email):
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    result = current_app.db.users.delete_one({"email": email})
    if result.deleted_count == 1:
        return jsonify({'message': 'User deleted successfully'}), 200
    else:
        return jsonify({'error': 'User not found'}), 404


@auth_.route('/forget-password-check', methods=["POST"])
def forget_password_check():
    email = request.form['email']
    password = request.form['password']
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    exists = bool(current_app.db.users.find_one({"email": email}))
    if exists:
        current_app.db.users.update_one(
            {"email": email},
            {"$set": {"password": password_hash}}
        )
        return jsonify({'result': 'success', 'msg': 'Password successfully changed!'})

    return jsonify({'result': 'failed', 'msg': 'Email does not match!'})


@auth_.route('/forget-password')
def forget_password():
    return render_template('auth/forget-password.html')


@auth_.route("/logout", methods=["DELETE"])
@token_required
def logout():
    try:
        response = {"message": "Token successfully deleted"}
        resp = make_response(jsonify(response))
        resp.set_cookie("mytoken", "", expires=0, path="/")
        return resp
    except (jwt.ExpiredSignatureError, jwt.exceptions.DecodeError):
        response = {"message": "Invalid token"}
        return jsonify(response), 401
