# Foodee (Version 3.5)

## Overview

Foodee is an AI-powered meal planning and nutrition platform that generates personalized meal suggestions, recipes, shopping lists, and nutritional analysis based on users' dietary preferences, restrictions, and meal history. The platform leverages OpenAI's GPT-4 models for intelligent meal planning and the Perplexity API for real-time price estimations.

## Features

### Core Features
- **Personalized Meal Planning**
  - AI-powered meal suggestions considering dietary restrictions and preferences
  - Cultural and dietary diversity support
  - Meal history tracking to ensure variety
  - Session-based meal planning for consistent user experiences

### Recipe & Nutrition
- **Comprehensive Recipe Generation**
  - Detailed cooking instructions and ingredient lists
  - Preparation and cooking time estimates
  - Difficulty level assessments
  - Cooking tips and variations
- **Nutritional Analysis**
  - Detailed macro and micronutrient breakdown
  - Per-serving nutritional information
  - Allergen identification
  - Dietary tags and classifications

### Shopping & Budgeting
- **Smart Shopping Lists**
  - AI-generated shopping lists based on meal plans
  - Intelligent ingredient aggregation
  - Store-friendly quantity suggestions
  - Real-time price estimations
  - Total meal cost calculations

### API & Security
- **RESTful API Integration**
  - FastAPI-based endpoints
  - JWT authentication
  - Rate limiting
  - Automatic key rotation
  - Prometheus metrics

## Technical Stack

- **Backend**: Python 3.8+, FastAPI
- **AI/ML**: OpenAI GPT-4, Perplexity API
- **Database**: MongoDB
- **Authentication**: JWT with automatic key rotation
- **Monitoring**: Prometheus metrics
- **Documentation**: Swagger/OpenAPI
- **Testing**: Jest, Mocha

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/foodee.git
cd foodee
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```ini
MONGO_DB_CONN_STRING=your_mongodb_connection_string
OPENAI_API_KEY=your_openai_api_key
Perplexity_API_KEY=your_perplexity_api_key
TEAMS_WEBHOOK_URL=your_teams_webhook_url
JWT_SECRET_KEY=your_jwt_secret_key
```

## Running the Application

### Key Scripts and Commands

1. Start the API server:
```bash
python fast_api.py
# API will be available at http://localhost:1233
```

2. Generate meal plans:
```bash
# For single user
python meal_generators_workingclass/openai_meal_gen_single_meal.py 1 1 username

# For multiple users
python meal_generators_workingclass/openai_meal_gen_single_meal.py 1 2
```

3. Run the orchestrator (complete process):
```bash
python eng/orch_sng_ml_wc.py
```

4. Generate shopping lists:
```bash
python eng/generate_shopping_list.py
```

5. Get price estimates:
```bash
python eng/perplexity_find_prices.py
```

6. Rotate JWT keys:
```bash
python eng/key_gen.py --rotate-now  # One-time rotation
python eng/key_gen.py --schedule 24  # Schedule rotation every 24 hours
```

### Security Management

1. Generate a new JWT secret key:
```bash
python eng/key_gen.py --rotate-now
```

2. Set up automated key rotation:
```bash
python eng/key_gen.py --schedule 168  # Weekly rotation
```

## API Documentation

Access the Swagger documentation at `http://localhost:1233/docs` after starting the API server.

Key endpoints:
- `/api/meal-plan`: Create new meal plans
- `/api/meal-plan/{username}/status`: Check meal plan status
- `/api/meal-plan/{username}/results/{session_id}`: Get meal plan results
- `/metrics`: Prometheus metrics
- `/health`: Service health check

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT-4 models
- Perplexity for price estimation API
- MongoDB for data storage
- FastAPI for API framework