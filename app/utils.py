import requests
import json
import time
from functools import wraps
from .config import OPENWEATHERMAP_API_KEY, OLLAMA_URL, OLLAMA_API_KEY


# ----------------- Retry Decorator with Exponential Backoff -----------------
def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"⚠️ Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        print(f"❌ All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


@retry_with_backoff(max_retries=2, base_delay=0.5)
def _fetch_url(url, timeout=10):
    """Helper function to fetch URL with retry logic."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _fetch_url_safe(url, timeout=10, default=None):
    """
    Safely fetch URL with retry, returning default on failure.
    Does not raise exceptions - returns default value instead.
    """
    try:
        return _fetch_url(url, timeout)
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return default


def get_uv_index(lat, lon):
    """
    Fetches UV index from Open-Meteo API (free, no API key required).
    Returns the current UV index or 0 if fetch fails.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=uv_index_max&current=uv_index&timezone=auto&forecast_days=1"
        data = _fetch_url_safe(url, timeout=5)
        
        if data:
            # Try to get current UV index first, fallback to daily max
            current_uv = data.get("current", {}).get("uv_index")
            if current_uv is not None:
                print(f"✅ UV Index (current): {current_uv}")
                return round(current_uv, 2)
            
            daily_uv = data.get("daily", {}).get("uv_index_max", [])
            if daily_uv and len(daily_uv) > 0:
                print(f"✅ UV Index (daily max): {daily_uv[0]}")
                return round(daily_uv[0], 2)
        
        print("⚠️ UV index not found in response, using default 0")
        return 0
        
    except Exception as e:
        print(f"⚠️ Error fetching UV index: {e}")
        return 0


def get_weather(lat, lon):
    """
    Fetches weather, AQI, and UV data from multiple sources.
    Returns a unified dictionary with temp, uv, aqi, city, etc.
    
    Uses:
    - OpenWeatherMap for weather and AQI
    - Open-Meteo for UV index (free, no API key)
    """
    result = {}
    
    if not OPENWEATHERMAP_API_KEY:
        result.update({"weather_error": "Missing OPENWEATHERMAP_API_KEY"})
        # Still try to get UV index even without OWM key
        result.update({"uv_index": get_uv_index(lat, lon)})
        return result

    # 1. Current Weather (for Temp & City Name)
    try:
        url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        data_weather = _fetch_url_safe(url_weather, timeout=10, default={})
        
        if data_weather and "main" in data_weather:
            main = data_weather.get("main", {})
            sys = data_weather.get("sys", {})
            weather_info = data_weather.get("weather", [{}])[0]
            
            # Get sunrise/sunset for outdoor recommendations
            sunrise = sys.get("sunrise")
            sunset = sys.get("sunset")
            
            result.update({
                "city": data_weather.get("name", "Unknown"),
                "country": sys.get("country", ""),
                "temp_max": main.get("temp_max"),
                "temp_min": main.get("temp_min"),
                "current_temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "timezone": data_weather.get("timezone"),
                "weather_condition": weather_info.get("main", "Clear"),
                "weather_description": weather_info.get("description", ""),
                "weather_icon": weather_info.get("icon", "01d"),
                "sunrise": sunrise,
                "sunset": sunset,
                "wind_speed": data_weather.get("wind", {}).get("speed", 0),
                "clouds": data_weather.get("clouds", {}).get("all", 0)
            })
            print(f"✅ Weather data fetched for {result.get('city')}")
        else:
            error_msg = data_weather.get("message", "Failed to fetch weather data")
            print(f"⚠️ OWM Weather Error: {error_msg}")
            result.update({"weather_error": error_msg})
    except Exception as e:
        print(f"❌ Exception fetching weather: {e}")
        result.update({"weather_error": str(e)})

    # 2. Air Pollution (for AQI)
    try:
        url_aqi = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
        data_aqi = _fetch_url_safe(url_aqi, timeout=10, default={})

        if data_aqi and data_aqi.get("list"):
            # OWM returns AQI 1 (Good) to 5 (Very Poor).
            # We map this to US AQI (0-500) for the app's color logic.
            owm_aqi = data_aqi["list"][0]["main"]["aqi"]
            components = data_aqi["list"][0].get("components", {})
            
            aqi_map = {1: 20, 2: 80, 3: 120, 4: 180, 5: 250}
            mapped_aqi = aqi_map.get(owm_aqi, 100)
            
            # Find dominant pollutant (max value)
            dominant = max(components, key=components.get) if components else "unknown"

            result.update({
                "aqi": mapped_aqi,
                "dominant_pollutant": dominant,
                "owm_aqi_raw": owm_aqi
            })
            print(f"✅ AQI data fetched: {mapped_aqi} (dominant: {dominant})")
        else:
            print("⚠️ AQI data not available")
            result.update({"aqi": 0, "aqi_error": "Could not fetch AQI"})
    except Exception as e:
        print(f"❌ Exception fetching AQI: {e}")
        result.update({"aqi": 0, "aqi_error": str(e)})

    # 3. UV Index from Open-Meteo (free, no API key required)
    result.update({"uv_index": get_uv_index(lat, lon)})

    return result

# ----------------- Local LLM (Ollama) -----------------
def ask_llm(prompt: str, model: str = "gpt-oss:120b"):
    """
    Calls either a local Ollama server or Ollama Cloud API based on OLLAMA_URL.
    Handles bearer auth automatically and returns structured suggestions.
    """

    try:
        headers = {"Content-Type": "application/json"}

        # 🌐 Cloud API (https://ollama.com/api/chat)
        if "ollama.com/api/chat" in OLLAMA_URL:
            if not OLLAMA_API_KEY:
                print("❌ Missing OLLAMA_API_KEY for cloud request")
                return {"error": "Missing API key"}

            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }

        # 💻 Local Ollama (http://localhost:11434/api/generate)
        else:
            payload = {"model": model, "prompt": prompt, "stream": False}

        # Make request
        response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            print(f"❌ LLM request failed ({response.status_code}):", response.text)
            return {"error": response.text}

        data = response.json()

        # Extract raw text
        if "ollama.com/api/chat" in OLLAMA_URL:
            raw_text = (
                data.get("message", {}).get("content")
                or data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
        else:
            raw_text = data.get("response", "")

        raw_text = (raw_text or "").strip()

        if not raw_text:
            print("⚠️ Empty response from LLM:", data)
            return {"suggestions": []}

        # 🧠 Handle JSON string responses like '{"suggestions": [...]}'
        try:
            cleaned = raw_text.strip("` \n")
            parsed = json.loads(cleaned)
            if isinstance(parsed, str):  # If double-encoded JSON
                parsed = json.loads(parsed)

            if isinstance(parsed, dict) and "suggestions" in parsed:
                return parsed
        except Exception as e:
            print("⚠️ JSON parse failed:", e)

        # 🪶 Fallback: split bullet points or sentences
        suggestions = [
            line.strip("-• ").strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]
        return {"suggestions": suggestions[:5]}

    except Exception as e:
        print("❌ Exception in ask_llm:", e)
        return {"error": str(e)}
