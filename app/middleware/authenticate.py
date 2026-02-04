from functools import wraps
from flask import request, jsonify, current_app, redirect, url_for
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


def admin_required(f):
    """
    Decorator that requires the user to be logged in AND have admin role.
    Must be used after @token_required or handle token validation itself.
    """
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.cookies.get("mytoken")
        if not token:
            return redirect(url_for("auth.sign_in", msg="Please login first!"))
        
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            user = current_app.db.users.find_one({"email": payload["id"]})
            
            if not user:
                return jsonify({"error": "User not found!"}), 404
            
            # Check if user has admin role
            if user.get("role", "").lower() != "admin":
                return jsonify({"error": "Admin access required!"}), 403
                
        except ExpiredSignatureError:
            return redirect(url_for("auth.sign_in", msg="Login time has expired!"))
        except DecodeError:
            return redirect(url_for("auth.sign_in", msg="Please login first!"))
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        
        return f(*args, **kwargs)
    return decorator
