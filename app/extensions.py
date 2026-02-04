from flask_socketio import SocketIO
from pymongo import MongoClient
import gridfs

# Socket.IO instance
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading', message_queue=None)
# Mongo globals (lazy initialized)
mongo_client = None
mongo_db = None
fs = None


def init_mongo(app):
    """
    Initialize MongoDB and GridFS using Flask app config.
    This MUST be called inside create_app().
    """
    global mongo_client, mongo_db, fs

    mongo_url = app.config.get("MONGODB_URL")
    db_name = app.config.get("DBNAME")

    if not mongo_url or not db_name:
        raise RuntimeError("MONGODB_URL or DBNAME is not set in app.config")

    mongo_client = MongoClient(mongo_url)
    mongo_db = mongo_client[db_name]
    fs = gridfs.GridFS(mongo_db)

    print(f"[MongoDB] Connected to database: {mongo_db.name}")
