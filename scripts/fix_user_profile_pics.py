"""
Script to update existing users with old local profile picture paths
to use UI Avatars URL instead.
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME", "ICU")

if not MONGO_URI:
    print("Error: MONGODB_URL not found in environment variables")
    sys.exit(1)

import certifi
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]

def fix_profile_pictures():
    """Update users with old local paths to use UI Avatars."""
    
    # Find all users with old local path pattern
    old_path_patterns = [
        "src/images/profiles/",  # Old local path
        "profiles/default.png",  # If this doesn't exist in GridFS
    ]
    
    users_collection = db.users
    
    # Find users with old paths
    users_to_update = users_collection.find({
        "$or": [
            {"profilePict": {"$regex": "^src/"}},  # Old local paths
            {"profilePict": {"$regex": "^profiles/"}},  # Old relative paths
            {"profilePict": None},  # Null values
            {"profilePict": ""},  # Empty strings
        ]
    })
    
    updated_count = 0
    
    for user in users_to_update:
        username = user.get('username', 'User')
        email = user.get('email', '')
        
        # Generate UI Avatars URL based on username
        new_profile_pic = f"https://ui-avatars.com/api/?name={username}&background=0891b2&color=fff&size=128"
        
        # Update the user
        result = users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"profilePict": new_profile_pic}}
        )
        
        if result.modified_count > 0:
            updated_count += 1
            print(f"Updated: {username} ({email})")
    
    print(f"\nTotal users updated: {updated_count}")

if __name__ == "__main__":
    print("Fixing user profile pictures...")
    print("=" * 50)
    fix_profile_pictures()
    print("=" * 50)
    print("Done!")
