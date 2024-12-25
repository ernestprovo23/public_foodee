import os
import requests
import json
import time
from pymongo import MongoClient
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables explicitly by specifying the path
load_dotenv()

# Access and validate environment variables
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
EDAMAM_APP_ID = os.getenv('EDAMAM_APP_ID')
EDAMAM_API_KEY = os.getenv('EDAMAM_API_KEY')
EDAMAM_USER_ID = "dataexperts101"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Add your OpenAI API key here

# Validate environment variables
if not MONGO_CONN_STRING or not EDAMAM_APP_ID or not EDAMAM_API_KEY or not EDAMAM_USER_ID or not OPENAI_API_KEY:
    raise ValueError(
        "All environment variables for MongoDB, Edamam API credentials, user ID, and OpenAI API key are required.")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# MongoDB Connection (Connection is pooled and reused across the script)
try:
    client = MongoClient(MONGO_CONN_STRING)
    db = client.foode
    print(f"Successfully connected to MongoDB at {MONGO_CONN_STRING}")

    log_collection = db.logs
    results_collection = db.results  # source
    recipes_collection = db.recipes  # destination
    users_summary_collection = db.users_summary

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

# Define logging with RotatingFileHandler
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
mongo_handler = logging.StreamHandler()  # You can also log to MongoDB or files
file_handler = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=5)  # 5MB per file, 5 backups

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mongo_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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


def simplify_meal_name(meal_name):
    """Use OpenAI API to simplify or split complex meal names into simpler terms."""

    messages = [
        {
            "role": "system",
            "content": (
                "Break down the following meal name into simpler, individual dishes or components suitable for "
                "recipe searches. Provide the components as a list of strings in the 'components' field."
            ),
        },
        {"role": "user", "content": f"'{meal_name}'"},
    ]

    # Define the Pydantic model for the expected response
    class SimplifiedMealTerms(BaseModel):
        components: List[str]

    # Initialize the OpenAI client with your API key
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Use an appropriate model that supports response_format
            messages=messages,
            max_tokens=300,
            temperature=0.3,
            n=1,
            response_format=SimplifiedMealTerms,
        )
        # Access the parsed components
        simplified_terms = response.choices[0].message.parsed.components
        logger.debug(f"Simplified meal terms for '{meal_name}': {simplified_terms}")
        return simplified_terms
    except Exception as e:
        logger.error(f"Error simplifying meal name '{meal_name}': {e}")
        return [meal_name]  # Fallback to the original name


# --- Error-handling decorators ---
def error_handling(func):
    """A decorator for global error handling with logging."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper


# Define the structured output schema for the recipes
class Recipe(BaseModel):
    label: str
    source: str
    url: str
    yield_: Optional[int] = None  # 'yield' is a reserved keyword
    ingredientLines: List[str]
    totalTime: Optional[float] = None
    calories: Optional[float] = None


class MealRecipe(BaseModel):
    name: str
    recipes: List[Recipe]  # List of recipes


class SideRecipe(BaseModel):
    name: str
    recipes: List[Recipe]  # List of recipes


class MealWithSides(BaseModel):
    meal: MealRecipe
    sides: List[SideRecipe]
    datetimeimported: datetime


# --- Edamam API Service ---
class EdamamAPIService:
    """
    This service interacts with the Edamam API to fetch recipes based on meal names.
    """

    def __init__(self, app_id, app_key, user_id, health_labels=None, excluded_ingredients=None, max_total_recipes=10,
                 testing_mode=True):
        self.app_id = app_id
        self.app_key = app_key
        self.user_id = user_id
        self.endpoint = "https://api.edamam.com/api/recipes/v2"
        self.max_requests_per_minute = 10
        self.sleep_time = 60 / self.max_requests_per_minute
        self.num_requests = 0
        self.max_total_recipes = max_total_recipes  # Max total recipes to fetch during testing
        self.testing_mode = testing_mode  # If True, limit the number of recipes and rate
        self.total_recipes_fetched = 0  # Counter for total recipes fetched

        # Store health labels and excluded ingredients
        self.health_labels = health_labels or []
        self.excluded_ingredients = excluded_ingredients or []

    def log_api_call(self, params, response, error=None):
        """Log API calls with request parameters and response metadata."""
        log_entry = {
            "api": "Edamam",
            "timestamp": datetime.now(),
            "params": params,
            "response_status": response.status_code if response else None,
            "response_content": response.text if response else None,
            "error": str(error) if error else None
        }
        try:
            # Insert the log into MongoDB
            log_collection.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Failed to log API call: {e}")

    def make_api_request(self, params):
        """Make a request to the Edamam API with rate limiting and logging."""
        self.num_requests += 1
        if self.testing_mode and self.num_requests > self.max_requests_per_minute:
            logger.info("Sleeping to respect rate limit...")
            time.sleep(self.sleep_time)
            self.num_requests = 0
        try:
            headers = {
                'Edamam-Application-Id': self.app_id,
                'Edamam-Application-Key': self.app_key,
                'Edamam-Account-User': self.user_id
            }
            response = requests.get(self.endpoint, params=params, headers=headers)
            # Log the API call
            self.log_api_call(params, response)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Edamam API request failed: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception during API request: {e}")
            # Log the exception
            self.log_api_call(params, None, error=e)
            return None

    def get_recipes(self, query_terms, num_recipes=3):
        """Fetch recipes from Edamam API using simplified query terms."""
        recipes = []
        for query in query_terms:
            if self.testing_mode and self.total_recipes_fetched >= self.max_total_recipes:
                logger.info("Reached maximum total recipes limit during testing.")
                break

            params = {
                'type': 'public',
                'q': query,
                'app_id': self.app_id,
                'app_key': self.app_key,
                'from': 0,
                'to': num_recipes  # Fetch top 'num_recipes' recipes
            }

            # Add health labels to params
            if self.health_labels:
                params['health'] = self.health_labels

            # Add excluded ingredients to params
            if self.excluded_ingredients:
                params['excluded'] = self.excluded_ingredients

            data = self.make_api_request(params)
            if not data or 'hits' not in data or not data['hits']:
                logger.warning(f"No recipes found for query: {query}")
                continue

            for hit in data['hits']:
                if "recipe" in hit:
                    recipe_data = hit["recipe"]
                    try:
                        recipe = Recipe(
                            label=recipe_data.get('label', ''),
                            source=recipe_data.get('source', ''),
                            url=recipe_data.get('url', ''),
                            yield_=int(recipe_data.get('yield', 0)),
                            ingredientLines=recipe_data.get('ingredientLines', []),
                            totalTime=recipe_data.get('totalTime', None),
                            calories=recipe_data.get('calories', None)
                        )
                    except Exception as e:
                        logger.error(f"Error parsing recipe data: {e}")
                        continue
                    recipes.append(recipe)
                    self.total_recipes_fetched += 1
                    if self.testing_mode and self.total_recipes_fetched >= self.max_total_recipes:
                        logger.info("Reached maximum total recipes limit during testing.")
                        break
                else:
                    logger.warning(f"No recipe data in hit for query: {query}")

            if len(recipes) >= num_recipes:
                break  # Break if we have enough recipes

        if len(recipes) == 0:
            logger.warning(f"No recipes found for queries: {query_terms}")

        return recipes[:num_recipes]


# --- Recipe Gathering Service ---
class RecipeGatheringService:
    """
    This service fetches meal suggestions from MongoDB,
    uses the Edamam API to get recipes for meals and sides,
    and stores the results in MongoDB.
    """

    def __init__(self, username, db, session_id=None, max_recipes_per_item=3, testing_mode=True):
        self.username = username
        self.db = db
        self.results_collection = db.results
        self.recipes_collection = db.recipes
        self.users_summary_collection = db.user_summary
        self.session_id = session_id
        self.max_recipes_per_item = max_recipes_per_item
        self.testing_mode = testing_mode

        # Fetch user's dietary restrictions
        self.dietary_restrictions = self.fetch_user_dietary_restrictions()
        # Map dietary restrictions to health labels and excluded ingredients
        self.health_labels, self.excluded_ingredients = self.map_dietary_restrictions()

        self.edamam_service = EdamamAPIService(
            EDAMAM_APP_ID,
            EDAMAM_API_KEY,
            EDAMAM_USER_ID,
            health_labels=self.health_labels,
            excluded_ingredients=self.excluded_ingredients,
            max_total_recipes=50,  # Adjust as needed
            testing_mode=testing_mode
        )

    def fetch_user_dietary_restrictions(self):
        """Fetch the user's dietary restrictions from the database."""
        user_doc = self.users_summary_collection.find_one({'username': self.username})
        if not user_doc:
            logger.error(f"No user summary found for username: {self.username}")
            return []
        dietary_restrictions = user_doc.get('Dietary_Restrictions', [])
        return [restriction.lower() for restriction in dietary_restrictions]

    def map_dietary_restrictions(self):
        """Map dietary restrictions to Edamam API health labels and excluded ingredients."""
        health_label_mapping = {
            'gluten': 'gluten-free',
            'peanuts': 'peanut-free',
            'peanut': 'peanut-free',
            'tree nuts': 'tree-nut-free',
            'tree nut': 'tree-nut-free',
            'soy': 'soy-free',
            'wheat': 'wheat-free',
            'eggs': 'egg-free',
            'shellfish': 'shellfish-free',
            'pork': 'pork-free',
            'red meat': 'red-meat-free',
            'alcohol': 'alcohol-free',
            # Add more mappings as needed
        }

        health_labels = set()
        excluded_ingredients = set()

        for restriction in self.dietary_restrictions:
            # Map to health labels if possible
            if restriction in health_label_mapping:
                health_labels.add(health_label_mapping[restriction])
            else:
                # Use the restriction as an excluded ingredient
                excluded_ingredients.add(restriction)

        return list(health_labels), list(excluded_ingredients)

    @error_handling
    def fetch_meal_suggestions(self):
        """Fetch the latest meal suggestions for the user."""
        query = {'username': self.username}
        if self.session_id:
            query['session_id'] = self.session_id  # Filter by session_id

        meal_suggestions_doc = self.results_collection.find_one(
            query,
            sort=[('datetimeimported', -1)]
        )
        if not meal_suggestions_doc:
            logger.error(f"No meal suggestions found for user {self.username} with session_id: {self.session_id}")
            return None

        return remove_object_ids(meal_suggestions_doc)

    @error_handling
    def generate_recipes(self):
        """Generates recipes for meals and sides."""
        meal_suggestions_doc = self.fetch_meal_suggestions()
        if not meal_suggestions_doc:
            return

        # Extract meal suggestions
        content = meal_suggestions_doc.get('content', {})
        if isinstance(content, dict):
            meal_suggestions = content  # Already parsed
        else:
            try:
                meal_suggestions = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding meal suggestions content: {e}")
                return

        results = []

        for meal_type in ['breakfast', 'lunch', 'dinner']:
            suggestion = meal_suggestions.get(meal_type)
            if not suggestion:
                continue  # Skip if no suggestion for this meal_type
            meal_name = suggestion['meal']
            sides = suggestion.get('sides', [])

            logger.info(f"Processing {meal_name}")

            # Simplify meal name using OpenAI API
            simplified_meal_terms = simplify_meal_name(meal_name)
            logger.debug(f"Simplified meal terms: {simplified_meal_terms}")

            # Fetch recipes for simplified meal terms
            try:
                meal_recipes = self.edamam_service.get_recipes(
                    query_terms=simplified_meal_terms,
                    num_recipes=self.max_recipes_per_item
                )
            except Exception as e:
                logger.error(f"Exception occurred while searching recipes for: {meal_name}. Exception: {str(e)}")
                meal_recipes = []

            # Initialize list to store side recipes
            side_recipes_list = []

            # Processing sides
            for side in sides:
                if not side:
                    continue
                # logger.info(f"   Searching recipes for side: {side}")

                # Simplify side name using OpenAI API
                simplified_side_terms = simplify_meal_name(side)
                # logger.debug(f"Simplified side terms: {simplified_side_terms}")

                # Fetch recipes for simplified side terms
                try:
                    side_recipes = self.edamam_service.get_recipes(
                        query_terms=simplified_side_terms,
                        num_recipes=self.max_recipes_per_item
                    )
                    side_recipes_list.append(SideRecipe(
                        name=side,
                        recipes=side_recipes
                    ))
                except Exception as e:
                    logger.error(f"Exception occurred while searching recipes for: {side}. Exception: {str(e)}")
                    side_recipes_list.append(SideRecipe(
                        name=side,
                        recipes=[]
                    ))

            meal_with_sides = MealWithSides(
                meal=MealRecipe(
                    name=meal_name,
                    recipes=meal_recipes
                ),
                sides=side_recipes_list,
                datetimeimported=datetime.now()
            )

            results.append(meal_with_sides.dict())

        # Save to MongoDB collection
        if results:
            for result in results:
                result['username'] = self.username
                result['session_id'] = self.session_id
            self.recipes_collection.insert_many(results)
            logger.info("Recipes saved to MongoDB")
            # print(json.dumps(results, indent=2, default=str))
        else:
            logger.info("No recipes generated.")

    def run(self):
        self.generate_recipes()


def main():
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

    # Prompt for testing mode
    testing_mode_input = input("Enable testing mode? (yes/no): ").strip().lower()
    testing_mode = testing_mode_input in ['yes', 'y']

    # Initialize the RecipeGatheringService
    recipe_service = RecipeGatheringService(
        username=username,
        db=db,
        session_id=session_id,
        max_recipes_per_item=3,  # Adjust as needed
        testing_mode=testing_mode
    )

    print(f"\nGenerating recipes for user: {username} with session_id: {session_id}...")
    print(f"Testing mode is {'enabled' if testing_mode else 'disabled'}.\n")

    # Generate recipes
    recipe_service.run()

    print("Recipe generation completed.")


if __name__ == "__main__":
    main()
