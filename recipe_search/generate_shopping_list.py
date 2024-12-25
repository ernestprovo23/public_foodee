import os
import openai
from pymongo import MongoClient
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import requests
import asyncio
from abc import ABC, abstractmethod
from pydantic import BaseModel
import uuid

# Load environment variables from .env file
load_dotenv()

# MongoDB Connection
try:
    client = MongoClient(os.getenv('MONGO_DB_CONN_STRING'))
    db = client.foode
    log_collection = db.logs  # Define log_collection here
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

# --- Logging Setup ---
class MongoDBHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.mongo_client = MongoClient(os.getenv('MONGO_DB_CONN_STRING'))
        self.db = self.mongo_client.foode
        self.log_collection = self.db.logs

    def emit(self, record):
        try:
            log_entry = self.format(record)
            log_document = {
                "log": log_entry,
                "datetime": datetime.now()
            }
            self.log_collection.insert_one(log_document)
        except Exception as e:
            print(f"Error writing to MongoDB log: {e}")

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
mongo_handler = MongoDBHandler()
file_handler = logging.FileHandler('shopping_list_service.log')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mongo_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(mongo_handler)
logger.addHandler(file_handler)

# --- Helper Functions ---
def remove_object_ids(document):
    """Removes MongoDB-specific ObjectId fields."""
    if isinstance(document, list):
        for item in document:
            item.pop('_id', None)
    else:
        document.pop('_id', None)
    return document

# --- Error-handling decorators ---
def error_handling(func):
    """A decorator for global error handling with logging."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper

# Define the structured output schema for the shopping list
class ShoppingListItem(BaseModel):
    category: str
    items: list[str]

class ShoppingListResponse(BaseModel):
    shopping_list: list[ShoppingListItem]

# --- AI Provider Abstractions ---
class AIProvider(ABC):
    """Abstract Base Class for AI providers."""

    @abstractmethod
    async def generate_shopping_list(self, prompt):
        pass

class OpenAIProvider(AIProvider):
    """OpenAI GPT-4 Integration."""

    def __init__(self, api_key):
        openai.api_key = api_key

    async def generate_shopping_list(self, prompt):
        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": """You are a highly knowledgeable and practical sous chef and 
                    kitchen manager, tasked with creating organized shopping lists for a family of four (two adults, 
                    two children). Your job is to take meal plans, extract the necessary ingredients, and generate a 
                    detailed store shopping list. The shopping list should reflect realistic purchasing amounts that 
                    a person would buy at the store, rather than the exact cooking measurements in the recipe.

                    Here is how you should approach the task: 
                    1. Extract the meal and side dish information from the input document. 
                    3. Extract the ingredientLines nested data from the input document.
                    2. For each ingredient, convert the recipe amounts into common store-friendly quantities or packaging sizes. For example: 
                       - If the recipe calls for '500g of chicken,' suggest '1 pack of chicken breasts (about 500g)'. 
                       - If the recipe calls for '200g of spinach,' suggest '1 bag of spinach (about 200g)'. 
                       - Use your knowledge of common grocery store packaging to reason through what a person would buy (round up where necessary). 
                    3. Organize the shopping list into common grocery store sections, such as Produce, Meat/Protein, Dairy, Pantry Items, and Spices. 
                    4. Be sure to avoid redundant items and combine ingredients when necessary (e.g., if fruit appears in multiple meals, group them together). 
                    5. Exclude any pantry staples (like salt or olive oil) unless they appear as a specific purchase requirement.

                    Provide the shopping list in the following JSON format adhering to this schema:
                    {
                        "shopping_list": [
                            {
                                "category": "string",
                                "items": ["string", "string"]
                            }
                        ]
                    }
                    """},
                    {"role": "user", "content": prompt}
                ],
                response_format=ShoppingListResponse  # Structured output with Pydantic schema
            )

            full_response = response

            # Handle refusals
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                logger.error(f"Model refusal: {response.choices[0].message.refusal}")
                return None, response

            # Extract the parsed response
            shopping_list_response = response.choices[0].message.parsed

            return shopping_list_response, full_response

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

#################################### --- Shopping List Generation Service --- ##########################################

class ShoppingListService:
    """
    This service fetches meal suggestions from MongoDB, generates a shopping list using AI,
    and stores the results.
    """

    def __init__(self, username, db, session_id=None):
        self.username = username
        self.db = db
        self.results_collection = db.results
        self.shopping_list_collection = db.shopping_lists
        self.meal_suggestions_limit = 2  # Number of meal suggestions to fetch
        self.session_id = session_id  # Include session_id

        # Get environment variables
        self.mongo_conn_string = os.getenv('MONGO_DB_CONN_STRING')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.teams_webhook_url = os.getenv('TEAMS_WEBHOOK_URL')

        # Verify environment variables are available
        missing_vars = []
        if not self.mongo_conn_string:
            missing_vars.append('MONGO_DB_CONN_STRING')
        if not self.openai_api_key:
            missing_vars.append('OPENAI_API_KEY')
        if not self.teams_webhook_url:
            missing_vars.append('TEAMS_WEBHOOK_URL')

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    async def fetch_recent_meal_suggestions(self):
        """Retrieve the recent meal suggestions from MongoDB."""
        query = {'username': self.username}
        if self.session_id:
            query['session_id'] = self.session_id  # Filter by session_id

        recent_meals = list(
            self.results_collection.find(query).sort([('_id', -1)])
        )
        if not recent_meals:
            logger.error(f"No recent meal suggestions found for username: {self.username} with session_id: {self.session_id}")
            return None
        return remove_object_ids(recent_meals)

    @error_handling
    async def generate_shopping_list_from_provider(self, provider: AIProvider, prompt: str):
        """Generate a shopping list using OpenAI GPT and save to MongoDB."""
        shopping_list_response, raw_response = await provider.generate_shopping_list(prompt)

        # Handle cases where the shopping list couldn't be generated
        if shopping_list_response is None:
            logger.warning(f"No shopping list generated for prompt: {prompt}")
            return

        # Log the full parsed response and raw API response
        logger.info(f"Full AI parsed response: {shopping_list_response.dict()}")
        logger.info(f"Full AI raw response: {raw_response}")

        # Access token usage and model information directly
        token_usage = raw_response.usage  # Access usage attribute
        model_info = raw_response.model  # Access model attribute

        try:
            # Create a serializable log_data object that excludes unencodable types
            log_data = {
                "username": self.username,
                "session_id": self.session_id,  # Include session_id
                "model": model_info,
                "structured_response": shopping_list_response.dict(),
                "completion_tokens": token_usage.completion_tokens if token_usage else None,
                "prompt_tokens": token_usage.prompt_tokens if token_usage else None,
                "total_tokens": token_usage.total_tokens if token_usage else None,
                "datetime": datetime.now()
            }

            # Insert the log data into the log_collection (MongoDB)
            log_collection.insert_one(log_data)

        except Exception as e:
            logger.error(f"Error logging data to MongoDB: {e}")

        # Save the parsed shopping list to the shopping_list_collection
        self.shopping_list_collection.insert_one({
            "username": self.username,
            "session_id": self.session_id,  # Include session_id
            "shopping_list": shopping_list_response.dict(),
            "datetimeimported": datetime.now()
        })

        print(shopping_list_response)
        return shopping_list_response

    async def run_shopping_list_generation(self):
        """Runs the shopping list generation process."""

        recent_meals = await self.fetch_recent_meal_suggestions()

        if not recent_meals:
            return  # Early return if no recent meals found

        # Build prompt based on recent meal suggestions
        prompt = self.build_prompt(recent_meals=recent_meals)

        # Create an instance of the AI Provider (OpenAI)
        openai_provider = OpenAIProvider(api_key=self.openai_api_key)

        # Generate the shopping list
        await self.generate_shopping_list_from_provider(openai_provider, prompt)

        # Log successful execution
        logger.info("Shopping list generation complete.")


    def build_prompt(self, recent_meals):
        """Constructs the required prompt for the AI to generate the shopping list."""
        # Extract meal contents from recent meals
        meal_contents = []
        for meal_doc in recent_meals:
            if 'content' in meal_doc:
                meal_contents.append(meal_doc['content'])

        # Build the prompt
        prompt = (
            "Here are the meal plans:\n"
            f"{json.dumps(meal_contents, indent=2)}\n\n"
            "Please generate a shopping list based on these meal plans."
        )

        return prompt


def get_latest_session_id(username, db):
    """
    Fetch the most recent session_id for the given username from the results_collection.

    Args:
        username (str): The username to search for.
        db (MongoClient.database): The MongoDB database instance.

    Returns:
        str or None: The latest session_id if found, else None.
    """
    latest_doc = db.results.find_one(
        {'username': username},
        sort=[('datetimeimported', -1)]
    )
    if latest_doc:
        session_id = latest_doc.get('session_id')
        logger.info(f"Latest session_id for user '{username}': {session_id}")
        return session_id
    else:
        logger.warning(f"No session_id found for user '{username}'.")
        return None


def validate_username(username, db):
    """Validate if the username exists in the users_summary_collection."""
    user_doc = db.user_summary.find_one({'username': username})
    if not user_doc:
        logger.error(f"Username '{username}' does not exist in the database.")
        print(f"Username '{username}' not found. Please try again.")
        return False
    return True


def validate_session_id(username, session_id, db):
    """Validate if the session_id exists for the given username in the results_collection."""
    doc = db.results.find_one({'username': username, 'session_id': session_id})
    if not doc:
        logger.error(f"Session ID '{session_id}' does not exist for user '{username}'.")
        print(f"Session ID '{session_id}' not found for user '{username}'. Please try again.")
        return False
    return True


async def main():
    # Prompt the user to input the username
    while True:
        username = input("Enter the username: ").strip()
        if validate_username(username, db):
            break
        else:
            continue

    # Ask if the user wants to use the latest session_id
    use_latest = input("Use the latest session_id? (yes/no): ").strip().lower()

    if use_latest in ['yes', 'y']:
        session_id = get_latest_session_id(username, db)
        if session_id:
            print(f"Using the latest session_id: {session_id}")
        else:
            print(f"No session_id found for user '{username}'. Please enter a specific session_id.")
            while True:
                session_id = input("Enter the session_id: ").strip()
                if validate_session_id(username, session_id, db):
                    break
                else:
                    continue
    else:
        # Prompt the user to input the specific session_id
        while True:
            session_id = input("Enter the session_id: ").strip()
            if validate_session_id(username, session_id, db):
                break
            else:
                continue

    shopping_list_service = ShoppingListService(username=username, db=db, session_id=session_id)

    print(f"Running shopping list generation for user: {username} with session_id: {session_id}...")

    # Run the shopping list generation
    await shopping_list_service.run_shopping_list_generation()


if __name__ == "__main__":
    asyncio.run(main())


