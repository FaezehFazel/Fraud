import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("AIzaSyCxNR-Kv9tSixvQUOcbOAKR43QQ8vJRPzI")

client = genai.Client(api_key=api_key)

for model in client.models.list():
    methods = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", None)
    print(model.name, methods)