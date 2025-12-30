import os
from dotenv import load_dotenv

load_dotenv()
# load_dotenv(override=True)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    TOKEN_KEY = os.getenv("TOKEN_KEY")
    MONGODB_URL = os.getenv("MONGODB_URL")  
    DBNAME = os.getenv("DB_NAME")
