import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

def get_business_ai_analysis():
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {
                "role": "user",
                "content": """You are a business analyst designed to understand how AI technology could be used across 
                large corporations. Provide me your overall plan for approaching work during initial meetings."""
            }
        ]
    )
    return response.choices[0].message.content

# Execute the function and print the response
if __name__ == "__main__":
    try:
        analysis = get_business_ai_analysis()
        print(analysis)
    except Exception as e:
        print(f"An error occurred: {str(e)}")