import os
from dotenv import load_dotenv

load_dotenv()

# External API keys
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Ollama settings
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")