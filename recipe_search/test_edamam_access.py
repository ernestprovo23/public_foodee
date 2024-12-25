import requests

def test_edamam_api():
    url = "https://api.edamam.com/api/recipes/v2"
    params = {
        "app_id": "0a52a1e7",
        "app_key": "c6258422273f5e8c391776fb7879f29d",
        "type": "public",
        "q": "chicken"  # Example query
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("Connection to Edamam API successful.")
        print("Response: Good to Go")
    else:
        print(f"Failed to connect to Edamam API. Status code: {response.status_code}")

if __name__ == "__main__":
    test_edamam_api()
