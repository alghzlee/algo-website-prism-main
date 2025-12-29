from functools import wraps
from flask import request, jsonify, current_app
import jwt
from flask_socketio import disconnect
from jwt import ExpiredSignatureError, DecodeError

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.cookies.get("mytoken")
        if not token:
            disconnect()
            return jsonify({"error": "Token is missing!"}), 401
        
        try:
            jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        except ExpiredSignatureError:
            disconnect()
            return jsonify({"error": "Token has expired!"}), 401
        except DecodeError:
            disconnect()
            return jsonify({"error": "Token is invalid!"}), 401
        except Exception as e:
            disconnect()
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        
        return f(*args, **kwargs)
    return decorator
