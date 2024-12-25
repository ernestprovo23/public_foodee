import asyncio
import logging
from pathlib import Path
import os
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
import json
from pymongo import MongoClient
import pymongo
from dotenv import load_dotenv
import sys


def setup_environment():
    # Get the project root directory (FoodE directory)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'

    print(f"Looking for .env at: {env_path.absolute()}")
    print(f".env exists: {env_path.exists()}")

    # Update Python path to include all necessary directories
    sys.path.extend([
        str(project_root / 'recipe_search'),
        str(project_root / 'meal_generators_workingclass'),
        str(project_root / 'pricing'),
        str(project_root / 'comms')
    ])

    if env_path.exists():
        load_dotenv(env_path)
        print("\nEnvironment variables after loading:")
        required_vars = ['MONGO_DB_CONN_STRING', 'OPENAI_API_KEY', 'TEAMS_WEBHOOK_URL', 'Perplexity_API_KEY']
        for var in required_vars:
            print(f"{var}: {'Set' if os.getenv(var) else 'Missing'}")
    else:
        raise FileNotFoundError(f"Environment file not found at {env_path}")

    required_vars = ['MONGO_DB_CONN_STRING', 'Perplexity_API_KEY', 'OPENAI_API_KEY', 'TEAMS_WEBHOOK_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('foodee_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Foodee.Orchestrator")

try:
    setup_environment()
except Exception as e:
    logger.error(f"Environment setup failed: {e}")
    raise


def verify_mongodb_connection(conn_string: str) -> bool:
    """Verify MongoDB connection and required collections exist."""
    try:
        client = MongoClient(conn_string)
        db = client.foode

        # List of required collections
        required_collections = [
            'results',
            'messages_outbound',
            'shopping_lists_with_prices',
            'recipe_nutrition_info',
            'formatted_nutrition_messages'
        ]

        # Get list of existing collections
        existing_collections = db.list_collection_names()

        # Check each required collection
        for collection in required_collections:
            if collection not in existing_collections:
                print(f"Warning: Required collection '{collection}' not found in database")
                return False

        # Test basic operations
        test_result = db.results.find_one()
        if test_result is None:
            print("Warning: Unable to read from results collection")
            return False

        return True

    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        return False
    finally:
        client.close()

@dataclass
class SessionInfo:
    username: str
    session_id: str
    iteration: int
    timestamp: datetime


class FoodeeSessionManager:
    def __init__(self, storage_path: str = "session_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.active_sessions: List[SessionInfo] = []

    def save_session(self, session: SessionInfo):
        self.active_sessions.append(session)
        self._write_sessions()

    def get_sessions_for_user(self, username: str) -> List[SessionInfo]:
        return [s for s in self.active_sessions if s.username == username]

    def _write_sessions(self):
        session_file = self.storage_path / "foodee_sessions.json"
        with open(session_file, 'w') as f:
            json.dump([{
                'username': s.username,
                'session_id': s.session_id,
                'iteration': s.iteration,
                'timestamp': s.timestamp.isoformat()
            } for s in self.active_sessions], f)


class FoodeeOrchestrator:
    def __init__(self):
        self.session_manager = FoodeeSessionManager()
        self.mongo_conn_string = os.getenv('MONGO_DB_CONN_STRING')
        self.perplexity_api_key = os.getenv('Perplexity_API_KEY')

        if not self.mongo_conn_string:
            raise ValueError("MongoDB connection string not found in environment variables")

        try:
            self.mongo_client = MongoClient(self.mongo_conn_string)
            self.db = self.mongo_client.foode
            self.results_collection = self.db.results
            logger.info(f"Successfully connected to MongoDB at {self.mongo_conn_string}")

            # Import the RecipeNutritionService class
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comms'))
                from recipe_nutrition import RecipeNutritionService
                self.RecipeNutritionService = RecipeNutritionService
            except Exception as e:
                logger.error(f"Failed to import RecipeNutritionService: {e}")
                raise

            test_count = self.results_collection.count_documents({})
            logger.info(f"Connected to results collection. Total documents: {test_count}")
            self.db.command('ping')

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise


    async def run_meal_generation(self, username: str, num_iterations: int):
        logger.info(f"Starting meal generation for user {username} with {num_iterations} iterations")

        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'meal_generators_workingclass'))
            from openai_meal_gen_single_meal import main as meal_main

            await meal_main(num_iterations=num_iterations, use_single_user='1', username=username)
            logger.info(f"Completed meal generation for user {username}")

        except Exception as e:
            logger.error(f"Error in meal generation: {e}")
            raise

    async def run_grocery_list_generation(self, username: str, session_ids: List[str]):
        logger.info(f"Generating grocery lists for user {username}")

        try:
            # Debug print environment variables
            print("Environment variables before import:")
            print(f"MONGO_DB_CONN_STRING: {'Set' if os.getenv('MONGO_DB_CONN_STRING') else 'Missing'}")
            print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
            print(f"TEAMS_WEBHOOK_URL: {'Set' if os.getenv('TEAMS_WEBHOOK_URL') else 'Missing'}")

            # Ensure environment variables are loaded
            project_root = Path(__file__).parent.parent
            load_dotenv(project_root / '.env')

            # Import after ensuring environment is set up
            from generate_shopping_list import ShoppingListService

            for session_id in session_ids:
                shopping_service = ShoppingListService(username=username, db=self.db, session_id=session_id)
                await shopping_service.run_shopping_list_generation()
                logger.info(f"Generated grocery list for session {session_id}")

        except Exception as e:
            logger.error(f"Error in grocery list generation: {e}")
            raise

    async def run_price_lookup(self, username: str, session_ids: List[str]):
        logger.info(f"Starting price lookup for user {username}")

        try:
            # Remove the sys.path.append and use direct import
            from perplexity_find_prices import ShoppingListPriceService

            for session_id in session_ids:
                price_service = ShoppingListPriceService(
                    username=username,
                    db=self.db,
                    session_id=session_id
                )
                result = price_service.generate_shopping_list_with_prices()
                if result:
                    logger.info(f"Generated prices for shopping list with session {session_id}")
                    logger.info(f"Total price for shopping list: ${result['total_price']:.2f}")
                else:
                    logger.warning(f"No shopping list found for session {session_id}")

        except Exception as e:
            logger.error(f"Error in price lookup: {e}")
            raise

    async def get_recent_session_ids(self, username: str, limit: int = 10) -> List[str]:
        try:
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    count = self.results_collection.count_documents({'username': username})
                    logger.info(f"Found {count} documents for user {username} in results collection")

                    if count == 0:
                        return []

                    results = list(self.results_collection.find(
                        {'username': username},
                        {'session_id': 1, 'datetimeimported': 1}
                    ).sort('datetimeimported', -1).limit(limit))

                    session_ids = []
                    for doc in results:
                        if 'session_id' in doc:
                            session_ids.append(doc['session_id'])
                        else:
                            logger.warning(f"Document missing session_id field: {doc}")

                    logger.info(f"Retrieved {len(session_ids)} session IDs from results collection")

                    sample_doc = self.results_collection.find_one({'username': username})
                    logger.info(f"Sample document structure: {sample_doc}")

                    return session_ids

                except pymongo.errors.AutoReconnect:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    logger.warning(f"Retrying MongoDB connection (attempt {attempt + 2}/{max_retries})")

                    self.mongo_client = MongoClient(self.mongo_conn_string)
                    self.db = self.mongo_client.foode
                    self.results_collection = self.db.results

            return []

        except Exception as e:
            logger.error(f"Error retrieving session IDs: {e}")
            logger.error(f"Exception details: {str(e)}")
            raise

    async def run_recipe_nutrition_generation(self, username: str, session_ids: List[str]):
        """Generate recipe and nutrition information for meals."""
        logger.info(f"Starting recipe and nutrition generation for user {username}")

        try:
            # Get the meal data for these specific session IDs
            for session_id in session_ids:
                logger.info(f"Processing nutrition for session ID: {session_id}")

                # First get the original meal data
                meal_doc = self.results_collection.find_one({
                    'username': username,
                    'session_id': session_id
                })

                if not meal_doc:
                    logger.warning(f"No meal data found for session {session_id}")
                    continue

                # Check if nutrition info already exists
                existing_nutrition = self.db.recipe_nutrition_info.find_one({
                    'username': username,
                    'session_id': session_id
                })

                if existing_nutrition:
                    logger.info(f"Nutrition info already exists for session {session_id}")
                    continue

                # Create nutrition service with the specific session
                nutrition_service = self.RecipeNutritionService(
                    username=username,
                    db=self.db,
                    session_id=session_id
                )

                # Get meal content from document
                if 'meal_type' not in meal_doc.get('content', {}):
                    logger.warning(f"Invalid meal content structure for session {session_id}")
                    continue

                logger.info(f"Processing meal: {meal_doc['content']['meal_type']}")

                # Call run_recipe_nutrition_generation without meal_data parameter
                result = await nutrition_service.run_recipe_nutrition_generation()

                if result:
                    logger.info(f"Generated recipe and nutrition info for session {session_id}")
                    logger.info(f"Original meal: {meal_doc['content']}")
                    logger.info(f"Generated nutrition: {result}")
                else:
                    logger.warning(f"Failed to generate nutrition info for session {session_id}")

            logger.info(f"Completed recipe and nutrition generation for user {username}")

        except Exception as e:
            logger.error(f"Error in recipe nutrition generation: {e}")
            raise

    async def read_nutrition_info(self, username: str, session_ids: List[str]):
        """Read and format nutrition information for meals in an email/text-friendly format."""
        logger.info(f"Reading nutrition information for user {username}")

        try:
            nutrition_collection = self.db.recipe_nutrition_info
            nutrition_docs = list(nutrition_collection.find({
                'username': username,
                'session_id': {'$in': session_ids}
            }).sort('datetime_generated', -1))

            if not nutrition_docs:
                logger.warning(f"No nutrition information found for user {username}")
                return

            # Initialize the formatted message
            formatted_message = f"🍽️ Recipe and Nutrition Guide for {username}\n\n"

            for doc in nutrition_docs:
                recipe_info = doc['recipe_nutrition_info']

                # Date and meal type header
                formatted_message += f"📅 {recipe_info['meal_date']} - {recipe_info['meal_type']}\n"
                formatted_message += "─" * 40 + "\n\n"

                # Main dish section
                main_dish = recipe_info['main_dish']
                formatted_message += f"🥘 Main Dish: {main_dish['name']}\n"
                formatted_message += f"⏱️ Prep: {main_dish['preparation_time_minutes']}min | "
                formatted_message += f"Cook: {main_dish['cooking_time_minutes']}min | "
                formatted_message += f"Difficulty: {main_dish['difficulty_level']}\n\n"

                # Ingredients
                formatted_message += "📝 Ingredients:\n"
                for ingredient in main_dish['ingredients']:
                    formatted_message += f"  • {ingredient}\n"
                formatted_message += "\n"

                # Main dish cooking steps
                formatted_message += "👩‍🍳 Cooking Instructions:\n"
                for step in main_dish['instructions']:
                    formatted_message += f"  {step['step_number']}. {step['instruction']}"
                    if step.get('time_minutes'):
                        formatted_message += f" ({step['time_minutes']} min)"
                    formatted_message += "\n"
                formatted_message += "\n"

                # Cooking tips
                if main_dish.get('tips'):
                    formatted_message += "💡 Chef's Tips:\n"
                    for tip in main_dish['tips']:
                        formatted_message += f"  • {tip}\n"
                    formatted_message += "\n"

                # Side dishes
                formatted_message += "🥗 Side Dishes:\n"
                for side in recipe_info['side_dishes']:
                    formatted_message += f"\n{side['name']}\n"
                    formatted_message += f"⏱️ Prep: {side['preparation_time_minutes']}min | "
                    formatted_message += f"Cook: {side['cooking_time_minutes']}min\n"

                    # Side dish ingredients
                    formatted_message += "Ingredients:\n"
                    for ingredient in side['ingredients']:
                        formatted_message += f"  • {ingredient}\n"

                    # Side dish instructions
                    formatted_message += "\nInstructions:\n"
                    for step in side['instructions']:
                        formatted_message += f"  {step['step_number']}. {step['instruction']}\n"
                    formatted_message += "\n"

                # Nutritional Information Summary
                formatted_message += "📊 Nutritional Information Summary\n"
                formatted_message += "─" * 40 + "\n\n"

                # Total meal nutrition
                total = recipe_info['total_meal_nutrients']
                formatted_message += "Complete Meal Totals:\n"
                formatted_message += f"• Calories: {total['calories']} kcal\n"
                formatted_message += f"• Protein: {total['protein_g']}g\n"
                formatted_message += f"• Carbohydrates: {total['carbohydrates_g']}g\n"
                formatted_message += f"• Fat: {total['fat_g']}g\n"
                formatted_message += f"• Fiber: {total['fiber_g']}g\n"
                formatted_message += f"• Sodium: {total['sodium_mg']}mg\n\n"

                # Dietary Information
                if total.get('dietary_tags'):
                    formatted_message += "🏷️ Dietary Tags: " + ", ".join(total['dietary_tags']) + "\n"
                if total.get('allergens') and total['allergens'] != ['None']:
                    formatted_message += "⚠️ Allergens: " + ", ".join(total['allergens']) + "\n"

                formatted_message += "\n" + "=" * 50 + "\n\n"

            # Store the formatted message in MongoDB for later use
            self.db.formatted_nutrition_messages.insert_one({
                'username': username,
                'session_ids': session_ids,
                'formatted_message': formatted_message,
                'datetime_generated': datetime.now()
            })

            return formatted_message

        except Exception as e:
            logger.error(f"Error formatting nutrition information: {e}")
            raise

    async def orchestrate_full_process(self, username: str, num_iterations: int):
        logger.info(f"Starting full Foodee process for user {username}")
        try:
            print("\n🚀 Starting full process")

            # Step 1: Generate meals with progress indication
            print(f"\n[1/7] Generating {num_iterations} meals...")
            await self.run_meal_generation(username, num_iterations)
            print("✓ Meal generation complete")
            await asyncio.sleep(5)

            # Step 2: Get most recent session IDs based on num_iterations
            print("\n[2/7] Retrieving session IDs...")
            session_ids = await self.get_recent_session_ids(username, num_iterations)

            if not session_ids:
                logger.warning(f"No session IDs found for user {username}")
                print("❌ No session IDs found")
                return

            print(f"\nProcessing {len(session_ids)} sessions:")
            for idx, sid in enumerate(session_ids, 1):
                print(f"{idx}. {sid}")

            try:
                # Step 3: Generate grocery lists
                print("\n[3/7] Generating grocery lists...")
                await self.run_grocery_list_generation(username, session_ids)
                print("✓ Grocery lists generated")
                await asyncio.sleep(5)

                # Step 4: Generate price lookups
                print("\n[4/7] Looking up prices...")
                await self.run_price_lookup(username, session_ids)
                print("✓ Price lookup complete")
                await asyncio.sleep(5)

                # Step 5: Format messages
                print("\n[5/7] Formatting messages...")
                await self.run_message_formatting(username, session_ids)
                await self.display_formatted_messages(username, session_ids)
                print("✓ Message formatting complete")

                # Step 6: Generate recipe and nutrition information
                print("\n[6/7] Generating recipe and nutrition information...")
                await self.run_recipe_nutrition_generation(username, session_ids)
                print("✓ Recipe and nutrition generation complete")
                await asyncio.sleep(5)

                # Step 7: Generate nutrition summary
                print("\n[7/7] Preparing nutrition summary...")
                nutrition_info = await self.read_nutrition_info(username, session_ids)
                if nutrition_info:
                    print("✓ Nutrition summary prepared")
                    print("\nNutrition information generated successfully!")

                print("\n✨ Full process completed successfully!")
                logger.info(f"Completed full Foodee process for user {username}")

            except Exception as e:
                print(f"\n❌ Error during process: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in orchestration process: {e}")
            print(f"\n❌ Error: {str(e)}")
            raise

    async def run_message_formatting(self, username: str, session_ids: List[str]):
        logger.info(f"Starting message formatting for user {username} with sessions {session_ids}")
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comms'))
            from format_message import process_user_documents

            # Debug print
            print(f"\nProcessing documents for sessions: {session_ids}")

            # Check if documents exist before processing
            doc_count = self.results_collection.count_documents({
                'username': username,
                'session_id': {'$in': session_ids}
            })

            if doc_count == 0:
                print(f"No documents found for sessions: {session_ids}")
                return

            # Pass both username and session_ids to process_user_documents
            process_user_documents(username, session_ids)
            print("Message formatting completed successfully")
            logger.info(f"Completed message formatting for user {username}")

        except Exception as e:
            logger.error(f"Error in message formatting: {e}")
            print(f"Error during message formatting: {str(e)}")
            raise

    async def display_formatted_messages(self, username: str, session_ids: List[str]):
        logger.info(f"Displaying formatted messages for user {username} with sessions {session_ids}")
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comms'))
            from outbound_messages import display_user_meal_plan

            # Debug print
            print(f"\nDisplaying meal plan for sessions: {session_ids}")

            # Check if formatted messages exist
            formatted_msgs = self.db.messages_outbound.find_one({
                'username': username,
                'session_ids': {'$in': session_ids}  # Add session_ids to query
            })

            if not formatted_msgs:
                print("No formatted messages found to display")
                return

            # Pass both username and session_ids to display_user_meal_plan
            display_user_meal_plan(username, session_ids)
            print("Message display completed successfully")
            logger.info(f"Completed displaying messages for user {username}")

        except Exception as e:
            logger.error(f"Error displaying messages: {e}")
            print(f"Error during message display: {str(e)}")
            raise


async def main():
    try:
        # Initial system checks
        print("Performing system checks...")

        # Check environment variables
        required_vars = ['MONGO_DB_CONN_STRING', 'Perplexity_API_KEY', 'OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("✓ Environment variables verified")

        # Check MongoDB connection
        try:
            client = MongoClient(os.getenv('MONGO_DB_CONN_STRING'))
            client.server_info()  # Will raise an exception if cannot connect
            print("✓ MongoDB connection successful")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to MongoDB: {str(e)}")
        finally:
            client.close()

        # Check required directories
        required_dirs = ['../meal_generators_workingclass', '../recipe_search', '../pricing', '../comms']
        for dir_path in required_dirs:
            full_path = os.path.join(os.path.dirname(__file__), dir_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        print("✓ Required directories verified")

        print("\nSystem checks completed successfully\n")

        # Get user input
        print("=== Foodee Meal Planning System ===\n")
        username = input("Enter username: ").strip()

        while True:
            try:
                num_iterations = int(input("Enter number of iterations (1-1000): "))
                if 1 <= num_iterations <= 1000:
                    break
                print("Number must be between 1 and 1000")
            except ValueError:
                print("Please enter a valid number")

        print("\nInitializing process...")
        orchestrator = FoodeeOrchestrator()

        # Run the main process
        await orchestrator.orchestrate_full_process(username, num_iterations)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        raise
    finally:
        if 'orchestrator' in locals():
            orchestrator.mongo_client.close()
            print("\nSession closed")


if __name__ == "__main__":
    try:
        # Set up asyncio policy for Windows if needed
        if os.name == 'nt':  # Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Create and get event loop
        loop = asyncio.get_event_loop()

        print("\n=== Foodee Meal Planning System ===\n")

        # Run the main function and wait for it to complete
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user")
        except Exception as e:
            print(f"\n\nError in main process: {str(e)}")
            logger.error(f"Error in main process: {e}", exc_info=True)
            raise
        finally:
            # Clean up
            try:
                pending = asyncio.all_tasks(loop)
                loop.run_until_complete(asyncio.gather(*pending))
            except Exception:
                pass
            loop.close()
            print("\nProcess completed")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)