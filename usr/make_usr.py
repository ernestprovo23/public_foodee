from pymongo import MongoClient
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv('MONGO_DB_CONN_STRING'))
db = client.foode

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_test_user():
    # Create a test user
    test_user = {
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": pwd_context.hash("testpassword"),
        "disabled": False
    }

    # Check if user exists
    existing_user = db.users.find_one({"username": test_user["username"]})
    if existing_user:
        print("Test user already exists")
        return

    # Insert user
    db.users.insert_one(test_user)
    print("Test user created successfully")

    # Create user summary
    user_summary = {
        "username": "testuser",
        "dietary_restrictions": [],
        "food_preferences": {}
    }

    # Insert user summary
    db.user_summary.insert_one(user_summary)
    print("User summary created successfully")


if __name__ == "__main__":
    create_test_user()