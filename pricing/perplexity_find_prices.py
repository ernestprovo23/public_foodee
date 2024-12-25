import os
import requests
import json
import re
import time
import logging
import backoff
from typing import Optional, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
PERPLEXITY_API_KEY = os.getenv('Perplexity_API_KEY')

# Validate environment variables
if not MONGO_CONN_STRING or not PERPLEXITY_API_KEY:
    raise ValueError("Environment variables for MongoDB and Perplexity API key are required.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_lookup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# MongoDB Connection
try:
    client = MongoClient(MONGO_CONN_STRING)
    db = client.foode
    logger.info(f"Successfully connected to MongoDB at {MONGO_CONN_STRING}")

    log_collection = db.logs
    shopping_list_collection = db.shopping_lists
    shopping_list_with_prices_collection = db.shopping_lists_with_prices

except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise


class PerplexityAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.logger = logging.getLogger(__name__)

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=3,
        max_time=300
    )
    def make_query(self, prompt: str, model: str = "llama-3.1-sonar-large-128k-online",
                   timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Make a query to the Perplexity API with exponential backoff retry logic"""
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
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
                    5. Exclude any pantry staples (like salt or olive oil) unless they appear as a specific purchase requirement ."""},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = self.session.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 524:  # Cloudflare timeout
                self.logger.warning(f"Cloudflare timeout encountered for prompt: {prompt}")
                return None
            raise
        except Exception as e:
            self.logger.error(f"Error in API request: {str(e)}")
            return None


class PriceLookupService:
    def __init__(self, api_key: str):
        self.perplexity_api = PerplexityAPI(api_key=api_key)
        self.price_cache = {}  # Simple cache to avoid repeated lookups
        self.logger = logging.getLogger(__name__)

    def _extract_price(self, text: str) -> float:
        """Enhanced price extraction with fallback patterns"""
        patterns = [
            r'\$(\d+(?:\.\d{2})?)',  # Standard price format ($XX.XX)
            r'(\d+(?:\.\d{2})?)\s*dollars',  # Price in words
            r'costs?\s*\$?(\d+(?:\.\d{2})?)',  # "costs $XX.XX"
            r'price[ds]?\s*(?:at|is|of)?\s*\$?(\d+(?:\.\d{2})?)',  # "priced at $XX.XX"
            r'(\d+(?:\.\d{2})?)',  # Any number as last resort
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue

        return 0.0

    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item names for better caching and comparison"""
        return ' '.join(item_name.lower().split())

    def get_item_price(self, item_name: str) -> float:
        """Get item price with caching and enhanced error handling"""
        normalized_name = self._normalize_item_name(item_name)

        # Check cache first
        if normalized_name in self.price_cache:
            self.logger.info(f"Cache hit for item: {item_name}")
            return self.price_cache[normalized_name]

        prompt = (
            f"What is the current average retail price of {item_name} in major US grocery stores? "
            f"Please respond with a precise price point based on current market data. "
            f"Format the response with the price in dollars, like this: 'The current average price is $X.XX'"
        )

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.perplexity_api.make_query(prompt)

                if response and 'choices' in response:
                    answer = response['choices'][0]['message']['content']
                    price = self._extract_price(answer)

                    if price > 0:
                        self.price_cache[normalized_name] = price
                        return price

                    self.logger.warning(f"Could not extract price from response for item: {item_name}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

            except Exception as e:
                self.logger.error(f"Error getting price for {item_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

        fallback_price = self._get_fallback_price(item_name)
        self.logger.warning(f"Using fallback price for {item_name}: ${fallback_price}")
        return fallback_price

    def _get_fallback_price(self, item_name: str) -> float:
        """Implement fallback pricing strategy when API fails"""
        item_lower = item_name.lower()

        # Define category-based fallback prices
        fallback_prices = {
            'meat': 12.99,
            'fish': 12.99,
            'seafood': 15.99,
            'vegetable': 3.99,
            'fruit': 3.99,
            'dairy': 4.99,
            'spice': 4.99,
            'seasoning': 4.99,
            'pantry': 5.99,
            'grain': 3.99,
            'bread': 3.99
        }

        # Check for category matches
        for category, price in fallback_prices.items():
            if category in item_lower:
                return price

        return 5.99  # Default fallback price

    def get_shopping_list_with_prices(self, shopping_list: list) -> list:
        """Process entire shopping list with improved error handling and logging"""
        shopping_list_with_prices = []
        total_items_processed = 0
        total_items = sum(len(category['items']) for category in shopping_list)

        for category_item in shopping_list:
            category = category_item['category']
            items = category_item['items']
            items_with_prices = []

            for item_name in items:
                total_items_processed += 1
                self.logger.info(f"Processing item {total_items_processed}/{total_items}: {item_name}")

                price = self.get_item_price(item_name)
                items_with_prices.append({
                    "name": item_name,
                    "price": round(price, 2)
                })

            shopping_list_with_prices.append({
                "category": category,
                "items": items_with_prices
            })

        return shopping_list_with_prices


class ShoppingListPriceService:
    def __init__(self, username: str, db, session_id: str = None):
        self.username = username
        self.db = db
        self.shopping_list_collection = db.shopping_lists
        self.price_lookup_service = PriceLookupService(api_key=PERPLEXITY_API_KEY)
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)

    def fetch_latest_shopping_list(self) -> Optional[Dict]:
        """Fetch the most recent shopping list for the user"""
        query = {'username': self.username}
        if self.session_id:
            query['session_id'] = self.session_id

        shopping_list_doc = self.shopping_list_collection.find_one(
            query,
            sort=[('datetimeimported', -1)]
        )

        if not shopping_list_doc:
            self.logger.error(f"No shopping list found for user {self.username} with session_id: {self.session_id}")
            return None

        return self._remove_object_ids(shopping_list_doc)

    def _remove_object_ids(self, document):
        """Removes MongoDB-specific ObjectId fields"""
        if isinstance(document, list):
            for item in document:
                item.pop('_id', None)
        else:
            document.pop('_id', None)
        return document

    def generate_shopping_list_with_prices(self) -> Optional[Dict]:
        """Generates the shopping list with prices"""
        shopping_list_doc = self.fetch_latest_shopping_list()
        if not shopping_list_doc:
            return None

        shopping_list = shopping_list_doc.get('shopping_list', {}).get('shopping_list', [])
        shopping_list_with_prices = self.price_lookup_service.get_shopping_list_with_prices(shopping_list)

        # Calculate total price
        total_price = sum(
            item["price"]
            for category in shopping_list_with_prices
            for item in category["items"]
        )

        result_doc = {
            "username": self.username,
            "session_id": self.session_id,
            "shopping_list_with_prices": shopping_list_with_prices,
            "total_price": round(total_price, 2),
            "datetimeimported": datetime.now()
        }

        # Save to database
        self.db.shopping_lists_with_prices.insert_one(result_doc)

        # Log the result
        self.logger.info(f"Generated shopping list with prices for user: {self.username}")
        print(json.dumps(result_doc, indent=2, default=str))

        return result_doc


def validate_username(username: str, db) -> bool:
    """Validate if the username exists in the users_summary_collection"""
    user_doc = db.user_summary.find_one({'username': username})
    if not user_doc:
        logger.error(f"Username '{username}' does not exist in the database.")
        print(f"Username '{username}' not found. Please try again.")
        return False
    return True


def validate_session_id(username: str, session_id: str, db) -> bool:
    """Validate if the session_id exists for the given username"""
    doc = db.shopping_lists.find_one({'username': username, 'session_id': session_id})
    if not doc:
        logger.error(f"Session ID '{session_id}' does not exist for user '{username}'.")
        print(f"Session ID '{session_id}' not found for user '{username}'. Please try again.")
        return False
    return True


def get_latest_session_id(username: str, db) -> Optional[str]:
    """Fetch the most recent session_id for the given username"""
    latest_doc = db.shopping_lists.find_one(
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


def main():
    # Prompt for username
    while True:
        username = input("Enter the username: ").strip()
        if validate_username(username, db):
            break

    # Ask about using latest session_id
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
        while True:
            session_id = input("Enter the session_id: ").strip()
            if validate_session_id(username, session_id, db):
                break

    # Create service and generate prices
    shopping_list_price_service = ShoppingListPriceService(
        username=username,
        db=db,
        session_id=session_id
    )

    print(f"Generating shopping list with prices for user: {username} with session_id: {session_id}...")
    shopping_list_price_service.generate_shopping_list_with_prices()


if __name__ == "__main__":
    main()