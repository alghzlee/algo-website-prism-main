from flask import Flask, request, render_template, current_app, Blueprint, jsonify, redirect, url_for
import pandas as pd
import jwt

home_ = Blueprint('home', __name__)

@home_.route('/')
def home():   
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        user_info = current_app.db.users.find_one({"email": payload["id"]})
        return render_template('home/bed-selection.html', user_info=user_info)
    except jwt.ExpiredSignatureError:
        return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
    except jwt.exceptions.DecodeError:
        return redirect(url_for("auth.sign_in", msg="Please login first!"))
    