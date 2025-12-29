from flask import Flask, request, render_template, current_app, Blueprint, jsonify, redirect, url_for
import jwt
from app.middleware.authenticate import token_required

similarity_ = Blueprint('similarity', __name__)

@similarity_.route('/similarity/<bed_id>')
def similarity(bed_id):
    myToken = request.cookies.get("mytoken")
    SECRET_KEY = current_app.config['SECRET_KEY']
    try:
        payload = jwt.decode(myToken, SECRET_KEY, algorithms=["HS256"])
        user_info = current_app.db.users.find_one({"email": payload["id"]})
        return render_template('dashboard/similarity.html', user_info=user_info, bed_id=bed_id)
    except jwt.ExpiredSignatureError:
        return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
    except jwt.exceptions.DecodeError:
        return redirect(url_for("auth.sign_in", msg="Please login first!"))
    
@similarity_.route('/get-data-similarity', methods=['GET'])
@token_required
def get_data_similarity():
    result = list(current_app.db.similarity_patient.find({}, {'_id' : False}))      
    return jsonify(result)

@similarity_.route('/search-data-similarity', methods=['POST'])
@token_required
def search_data_similarity():
    filters = request.form.to_dict()  
    query = {}

    if filters.get("gender"):
        query["gender"] = filters["gender"]  

    def add_range_filter(field_name):
        start_value = filters.get(f"{field_name}_start")
        end_value = filters.get(f"{field_name}_end")
        
        if start_value or end_value:
            range_query = {}
            if start_value:
                range_query["$gte"] = str(start_value)
            if end_value:
                range_query["$lte"] = str(end_value)
            query[field_name] = range_query

    add_range_filter("age")
    add_range_filter("weight")
    add_range_filter("height")

    try:
        result = list(current_app.db.similarity_patient.find(query, {"_id": False}))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error querying database: {str(e)}"}), 500


