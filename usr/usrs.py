import uuid
import os
import openai
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import random
from datetime import datetime, UTC
from typing import List
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize OpenAI API
client = openai.Client(api_key=OPENAI_API_KEY)

# MongoDB Connection
client_db = MongoClient(MONGO_CONN_STRING)
db = client_db.foode
user_summary_collection = db.user_summary

class UserSummary(BaseModel):
    username: str = Field(description="Generated username based on input or a creative food-related username")
    dietary_preferences: List[str]
    dietary_restrictions: List[str]
    food_tags: List[str]
    diet_label_tags: List[str]
    medical_conditions: List[str]
    age: int | None
    notes: str

def fetch_random_user_summaries(num_summaries=4):
    """Fetch a random set of user summary documents from MongoDB."""
    documents = list(user_summary_collection.find())
    random_summaries = random.sample(documents, min(num_summaries, len(documents)))

    for doc in random_summaries:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

    return "\n\n".join([json.dumps(doc, indent=4) for doc in random_summaries])


def read_user_data(file_name: str = "user_input.txt") -> str:
    """Read the user data from the specified file in the same directory as the script."""
    try:
        # Get the directory of the currently running script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one directory level to reach FoodE folder
        base_dir = os.path.dirname(script_dir)

        # Build the full file path
        file_path = os.path.join(base_dir, "usr", file_name)

        print(f"Attempting to read file from: {file_path}")  # Debug print

        # Open and read the file
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading user data file: {e}")
        print(f"Current working directory: {os.getcwd()}")  # Debug print
        return ""


def generate_user_summary():
    """Generate a structured user summary using file input."""
    try:
        # Get the user data from file
        user_data = read_user_data()
        if not user_data:
            raise ValueError("No user data found in input file")

        # Get random user summaries for context
        random_user_summaries = fetch_random_user_summaries(4)

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "From the perspective, paradigm and knowledge base of a PhD level expert in your field adopt the following persona: "
                        "You are an AI chef, nutritionist, and certified dietitian. Your tasks are the following : "
                        f"Take the following unstructured user input {user_data} "
                        "and make the best attempt at fitting it into a structured format into aggregate groups to user summary documents, including fields labeled 'username' "
                        "'dietary_preferences' (food or ingredients acceptable for the user), 'dietary_restrictions' (food the user absolutely cannot or will not consume), "
                        "'food_tags' (for general description or tags of types of ingredients included), 'diet_label_tags' (to display the users dietary types or label types), "
                        "and meta information such as age, medical history or medical conditions. See these other user summaries to get context of how we structured them for other users\n\n"
                        f"{random_user_summaries}"
                    )
                }
            ],
            response_format=UserSummary
        )

        return completion.choices[0].message.parsed.dict()

    except Exception as e:
        print(f"Error during user summary generation: {e}")
        return None

def main():
    """Main program function to generate a user summary and insert it into MongoDB."""
    structured_data = generate_user_summary()

    if structured_data:
        document = {
            **structured_data,  # Spread the structured data at root level
            "meta": {
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "change_id": str(uuid.uuid4()),
                "last_modified_by": "admin",
                "active_status": "active",
                "valid_from": datetime.now(UTC).isoformat(),
                "valid_to": 'null',
                "source_system": "source_system",
                "data_quality_score": 0.99
                }
        }

        try:
            result = user_summary_collection.insert_one(document)
            print("Document successfully inserted into MongoDB with ID:", result.inserted_id)
        except Exception as e:
            print(f"Error inserting document into MongoDB: {e}")
    else:
        print("Failed to generate the user summary document.")

if __name__ == "__main__":
    main()