import os
from dotenv import load_dotenv

load_dotenv()

# External API keys
WAQI_API_KEY = os.getenv("WAQI_API_KEY")

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
