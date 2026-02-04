import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug: Print environment check at startup
print("=" * 50, file=sys.stderr)
print("PRISM CONFIG - Environment Check", file=sys.stderr)
print("=" * 50, file=sys.stderr)
print(f"SECRET_KEY: {'SET' if os.getenv('SECRET_KEY') else 'MISSING!'}", file=sys.stderr)
print(f"TOKEN_KEY: {'SET' if os.getenv('TOKEN_KEY') else 'MISSING!'}", file=sys.stderr)
print(f"MONGODB_URL: {'SET' if os.getenv('MONGODB_URL') else 'MISSING!'}", file=sys.stderr)
print(f"DB_NAME: {'SET' if os.getenv('DB_NAME') else 'MISSING!'}", file=sys.stderr)
print(f"PORT: {os.getenv('PORT', '5001 (default)')}", file=sys.stderr)
print("=" * 50, file=sys.stderr)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-me")
    TOKEN_KEY = os.getenv("TOKEN_KEY", "default-token-key")
    MONGODB_URL = os.getenv("MONGODB_URL")
    DBNAME = os.getenv("DB_NAME")
    
    # Validate required env vars
    if not MONGODB_URL:
        print("ERROR: MONGODB_URL is required!", file=sys.stderr)
    if not DBNAME:
        print("ERROR: DB_NAME is required!", file=sys.stderr)
