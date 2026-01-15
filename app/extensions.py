from flask_socketio import SocketIO
from pymongo import MongoClient
import os

socketio = SocketIO(cors_allowed_origins="*", async_mode='eventlet', message_queue=None)

MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME")

if not MONGODB_URL or not DB_NAME:
    raise RuntimeError("MONGODB_URL or DB_NAME is not set")

mongo_client = MongoClient(MONGODB_URL)
mongo_db = mongo_client[DB_NAME]
