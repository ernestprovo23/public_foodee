import os
import openai
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import requests
from abc import ABC, abstractmethod
from pydantic import BaseModel
import time
import uuid
from collections import deque
import asyncio
import aiohttp
from pymongo import MongoClient
import sys
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access and validate environment variables
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TEAMS_WEBHOOK_URL = os.getenv('TEAMS_WEBHOOK_URL')

# MongoDB Connection (Connection is pooled and reused across the script)
client = MongoClient(MONGO_CONN_STRING)
db = client.foode
log_collection = db.logs
results_collection = db.results

if not MONGO_CONN_STRING or not OPENAI_API_KEY or not TEAMS_WEBHOOK_URL:
    raise ValueError("Environment variables for MongoDB, OpenAI API key, and Teams Webhook URL are required.")

# MongoDB Connection (Connection is pooled and reused across the script)
try:
    client = MongoClient(MONGO_CONN_STRING)
    db = client.foode
    print(f"Successfully connected to MongoDB at {MONGO_CONN_STRING}")

    log_collection = db.logs
    results_collection = db.results

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise


# Define a separate reusable Teams notification sending function
def send_message_to_teams(content):
    headers = {"Content-Type": "application/json"}
    response = requests.post(TEAMS_WEBHOOK_URL, headers=headers, data=json.dumps(content))
    if response.status_code != 200:
        raise ValueError(f"Request to Teams returned with status {response.status_code}: {response.text}")


# --- Logging Setup ---
class MongoDBHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_document = {
            "log": log_entry,
            "datetime": datetime.now()
        }
        log_collection.insert_one(log_document)


# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# MongoDB and File handlers for logging
mongo_handler = MongoDBHandler()

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
file_handler = logging.FileHandler(log_file_path)

# Formatters
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


# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    async def acquire(self):
        while True:
            now = time.monotonic()
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                break
            else:
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)


rate_limiter = RateLimiter(max_calls=4999, period=60)  # 1000 requests per 60 seconds


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


# Define the structured output schema
class MealSuggestion(BaseModel):
    meal: str
    sides: list[str]


class MealResponseSchema(BaseModel):
    meal_type: MealSuggestion


# --- AI Provider Abstractions ---
class AIProvider(ABC):
    """Abstract Base Class for AI providers."""

    @abstractmethod
    async def generate_meal_suggestions(self, prompt):
        pass


class OpenAIProvider(AIProvider):
    """OpenAI implementation for meal suggestion generation with enhanced diversity controls."""

    def __init__(self, api_key, username):
        openai.api_key = api_key
        self.username = username

    async def generate_meal_suggestions(self, prompt):
        await rate_limiter.acquire()
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)

            response = openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are an expert AI chef and nutritionist specializing in 
                    personalized meal planning. 
                    Your core mission is to create diverse, culturally rich meal suggestions that are:

                    1. DIVERSITY REQUIREMENTS:
                    - Never repeat the same protein type in consecutive suggestions
                    - Vary cooking methods (grilling, baking, stir-frying, etc.)
                    - Include dishes from different cultural backgrounds
                    - Alternate between complex and simple preparations
                    - Mix different flavor profiles (savory, spicy, umami, etc.)
                    
                    2. USER PROFILE INTEGRATION:
                    - Strictly adhere to all dietary restrictions in the user profile
                    - Consider cultural preferences and sensitivities
                    - Respect any health-related dietary needs
                    - Account for cooking skill level and time constraints
                    - Work within budget considerations for ingredients
                    
                    3. MEAL COMPOSITION RULES:
                    - Main dish must explicitly state the protein type
                    - Include 2-3 complementary side dishes that enhance the meal
                    - Ensure nutritional balance in the overall meal
                    - Consider seasonal availability of ingredients
                    - Provide varied textures in the meal components
                    
                    4. INNOVATION GUIDELINES:
                    - Suggest creative combinations while maintaining familiarity
                    - Incorporate fusion elements when appropriate
                    - Offer interesting variations on familiar dishes
                    - Consider modern dietary trends while respecting traditional preparations

                    Remember: Each meal suggestion should be completely different from previous ones, creating a rich, 
                    varied meal plan that keeps users engaged and satisfied.
                     
                     """},
                    {"role": "user", "content": prompt}
                ],
                response_format=MealResponseSchema,
                temperature=1
            )

            full_response = response
            print(f'{self.username} processed successfully...')

            meal_suggestions = response.choices[0].message.parsed

            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                logger.error(f"Model refusal: {response.choices[0].message.refusal}")
                return None, response

            return meal_suggestions, full_response

        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)


############################ --- Custom Meal Suggestion Service OpenAI GPT Built --- ###################################
class UserValidationError(Exception):
    """Custom exception for user validation errors"""
    pass


class UserValidator:
    """Handles user validation against the MongoDB database"""

    def __init__(self, db):
        self.db = db
        self.user_summary_collection = db.user_summary

    def validate_user(self, username: str) -> bool:
        """
        Validate if a user exists in the user_summary collection

        Args:
            username: The username to validate

        Returns:
            bool: True if user exists, False otherwise

        Raises:
            UserValidationError: If user doesn't exist
        """
        user = self.user_summary_collection.find_one({'username': username})
        if not user:
            raise UserValidationError(f"User '{username}' not found in user_summary collection")
        return True


class MealSuggestionService:
    """
    This service fetches user data, runs AI queries (concurrently if needed),
    stores the results, logs errors, and sends a summary to Microsoft Teams.
    """

    def __init__(self, num_days, username, db):
        self.num_days = num_days
        self.username = username
        self.db = db
        self.user_summary_collection = db.user_summary
        self.last_meal_query_limit = 10
        self.session_id = str(uuid.uuid4())  # Generate a unique session ID
        self.user_validator = UserValidator(db)

    async def validate_user(self):
        """Validate user exists before processing"""
        try:
            return self.user_validator.validate_user(self.username)
        except UserValidationError as e:
            logger.error(f"User validation failed: {str(e)}")
            raise


    async def fetch_user_summary(self):
        """Retrieve the user summary from MongoDB."""
        user_summary_data = self.user_summary_collection.find_one({'username': self.username})
        if not user_summary_data:
            logger.error(f"No user summary found for username: {self.username}")
            return None
        return remove_object_ids(user_summary_data)

    async def get_last_meals(self):
        """Retrieve the latest meal history."""
        last_meals = list(
            results_collection.find({'username': self.username}).sort([('_id', -1)]).limit(self.last_meal_query_limit))
        return remove_object_ids(last_meals)

    @error_handling
    async def generate_meals_from_provider(self, provider: AIProvider, prompt: str, collection):
        """Generate meal suggestions from OpenAI GPT using structured outputs and save to a specific MongoDB collection."""
        meal_suggestions, raw_response = await provider.generate_meal_suggestions(prompt)

        # Handle refusals, if present
        if meal_suggestions is None:
            # Log and handle the refusal case (no valid response)
            logger.warning(f"No meal suggestions generated for prompt: {prompt}")
            return

        # Log the full parsed response and raw API response
        logger.info(f"Full AI parsed response: {meal_suggestions}")  # Structured response in dict form
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
                "structured_response": meal_suggestions.dict(),  # Convert to dict
                "completion_tokens": token_usage.completion_tokens if token_usage else None,
                "prompt_tokens": token_usage.prompt_tokens if token_usage else None,
                "total_tokens": token_usage.total_tokens if token_usage else None,
                "datetime": datetime.now()
            }

            # Insert the log data into the log_collection (MongoDB)
            log_collection.insert_one(log_data)

        except Exception as e:
            logger.error(f"Error logging data to MongoDB: {e}")

        # Save the parsed meal suggestions content to the results collection
        collection.insert_one({
            "username": self.username,
            "session_id": self.session_id,  # Include session_id
            "content": meal_suggestions.dict(),  # Convert to dict
            "datetimeimported": datetime.now()
        })

        # print(meal_suggestions)
        return meal_suggestions

    async def run_meal_generation(self):
        """Runs AI generation workflows concurrently across multiple AI providers."""
        try:
            # Validate user first
            await self.validate_user()

            user_summary = await self.fetch_user_summary()
            last_meals = await self.get_last_meals()

            if not user_summary:
                logger.error(f"No user summary found for username: {self.username}")
                return  # Early return if no user summary found

            # Build prompt based on user summary and their meal history
            prompt = self.build_prompt(user_summary=user_summary, last_meals=last_meals)

            # Create instances of both AI Providers (OpenAI)
            openai_provider = OpenAIProvider(api_key=OPENAI_API_KEY, username=self.username)

            # Run both AI providers concurrently using asyncio.
            await asyncio.gather(
                self.generate_meals_from_provider(openai_provider, prompt, results_collection)
            )

        except UserValidationError as e:
            logger.error(f"User validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in meal generation: {str(e)}")
            raise

    def build_prompt(self, user_summary, last_meals):
        """Constructs a detailed prompt with enhanced diversity controls and user context."""

        # Extract recent meal history
        recent_meals = [
                           meal['content']['meal_type']['meal']
                           for meal in last_meals
                           if 'content' in meal
                              and 'meal_type' in meal['content']
                              and 'meal' in meal['content']['meal_type']
                       ][:5]  # Only look at last 5 meals for diversity check

        # Extract dietary preferences and restrictions
        dietary_restrictions = user_summary.get('Dietary_Restrictions', [])
        food_tags = user_summary.get('Food_Tags', {})

        prompt = f"""Generate a unique meal suggestion for user {user_summary} that is completely different from 
            their recent meals:

            RECENT MEALS (to avoid repetition):
            {json.dumps(recent_meals, indent=2)}
        
            USER PROFILE:
            Dietary Restrictions: {json.dumps(dietary_restrictions, indent=2)}
            Food Preferences: {json.dumps(food_tags, indent=2)}
        
            REQUIREMENTS:
            1. Generate ONE main dish with 2 complementary sides
            2. Main dish MUST explicitly state the protein type
            3. Ensure NO repetition of proteins or cooking styles from recent meals
            4. Consider user's dietary restrictions and preferences
            5. Create a balanced, nutritious meal combination
            6. Suggest practical dishes within a moderate to lower tier budget
        
            Note: Focus on generating the meals for users' preferences without their restrictions and respecting any 
            notes about their food interactions. The goal is to create the dish that meets the users' backgrounds.

        """

        return prompt


async def process_user(username, num_days, db):
    try:
        print(f"Processing user: {username}")
        # Create validator instance
        validator = UserValidator(db)

        # Validate user before creating service
        validator.validate_user(username)

        meal_service = MealSuggestionService(num_days=num_days, username=username, db=db)
        await meal_service.run_meal_generation()

    except UserValidationError as e:
        logger.error(f"Failed to process user: {str(e)}")
        print(f"Error: {str(e)}")
        return
    except Exception as e:
        logger.error(f"Error processing user {username}: {str(e)}")
        print(f"Error processing user {username}: {str(e)}")
        return

# Define the async main function
async def main(num_iterations: int, use_single_user: str, username: str = None):
    try:
        if use_single_user == '1' and username:
            # Validate single user before processing
            validator = UserValidator(db)
            validator.validate_user(username)
            usernames = [username]
        elif use_single_user == '2':
            users_collection = db.user_summary
            usernames = list(users_collection.distinct('username'))
            if not usernames:
                logger.error("No users found in user_summary collection")
                print("No users found in database.")
                return
            print(f"Found {len(usernames)} users in the database.")
        else:
            print("Invalid input. Please enter '1' or '2'.")
            return

        num_days = 1

        async def process_batch(batch):
            async with aiohttp.ClientSession() as session:
                tasks = [process_user(username, num_days, db) for username in batch]
                await asyncio.gather(*tasks)

        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")

            # Process users in batches of 10
            batch_size = 10
            for i in range(0, len(usernames), batch_size):
                batch = usernames[i:i + batch_size]
                await process_batch(batch)

            print(f"Completed iteration {iteration + 1}/{num_iterations}")

    except UserValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        print(f"Error: {str(e)}")
        return
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python openai_meal_gen.py <num_iterations> <use_single_user (1 or 2)> [username]")
        sys.exit(1)

    num_iterations = int(sys.argv[1])
    use_single_user = sys.argv[2]
    username = sys.argv[3] if len(sys.argv) > 3 else None

    if 1 <= num_iterations <= 1000:
        asyncio.run(main(num_iterations, use_single_user, username))
    else:
        print("Please enter a number between 1 and 1000.")