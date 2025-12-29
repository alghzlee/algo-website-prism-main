from flask import Flask, request, render_template, current_app, Blueprint, jsonify, redirect, url_for
import jwt
from app.middleware.authenticate import token_required

summary_ = Blueprint('summary', __name__)

@summary_.route('/summary/<bed_id>/<stay_id>')
def summary(bed_id, stay_id):
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        user_info = current_app.db.users.find_one({"email": payload["id"]})
        return render_template('dashboard/summary.html', user_info=user_info, bed_id=bed_id, stay_id=stay_id)
    except jwt.ExpiredSignatureError:
        return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
    except jwt.exceptions.DecodeError:
        return redirect(url_for("auth.sign_in", msg="Please login first!"))
    
@summary_.route('/get-summary/<stay_id>', methods=["GET"])
@token_required
def get_summary_data(stay_id):
    try:
        if not stay_id:
            return jsonify({"error": "Stay ID is required"}), 400

        result = current_app.db.similarity_patient.find_one({"stay_id": stay_id },{"_id": False})

        if not result:
            return jsonify({"error": "Data not found for the given Stay ID"}), 404

        return jsonify(result), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching data: {e}")
        return jsonify({"error": "An error occurred while fetching data"}), 500 