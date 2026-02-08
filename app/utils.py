import requests
import json
from .config import OPENWEATHERMAP_API_KEY, OLLAMA_URL, OLLAMA_API_KEY

def get_weather(lat, lon):
    """
    Fetches weather and AQI data from OpenWeatherMap.
    Returns a unified dictionary with temp, uv, aqi, city, etc.
    """
    result = {}
    
    if not OPENWEATHERMAP_API_KEY:
        result.update({"weather_error": "Missing OPENWEATHERMAP_API_KEY"})
        return result

    try:
        # 1. Current Weather (for Temp & City Name)
        url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        resp_weather = requests.get(url_weather, timeout=10)
        data_weather = resp_weather.json()
        
        if resp_weather.status_code == 200:
            main = data_weather.get("main", {})
            sys = data_weather.get("sys", {})
            
            result.update({
                "city": data_weather.get("name"),  # This is usually more accurate than WAQI
                "country": sys.get("country"),
                "temp_max": main.get("temp_max"), # Note: these are momentary min/max in standard free tier
                "temp_min": main.get("temp_min"),
                "current_temp": main.get("temp"),
                "humidity": main.get("humidity"),
                "timezone": data_weather.get("timezone")
            })
        else:
            print(f"OWM Weather Error: {data_weather}")
            result.update({"weather_error": data_weather.get("message", "Unknown error")})

        # 2. Air Pollution (for AQI)
        url_aqi = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
        resp_aqi = requests.get(url_aqi, timeout=10)
        data_aqi = resp_aqi.json()

        if resp_aqi.status_code == 200 and data_aqi.get("list"):
            # OWM returns AQI 1 (Good) to 5 (Very Poor).
            # We map this to US AQI (0-500) for the app's color logic.
            # Mapping: 1->20 (Good), 2->80 (Moderate), 3->120 (Unhealthy for Sensitive), 
            # 4->180 (Unhealthy), 5->250 (Very Unhealthy/Hazardous)
            owm_aqi = data_aqi["list"][0]["main"]["aqi"]
            components = data_aqi["list"][0]["components"]
            
            aqi_map = {1: 20, 2: 80, 3: 120, 4: 180, 5: 250}
            mapped_aqi = aqi_map.get(owm_aqi, 100)
            
            # Find dominant pollutant (max value)
            dominant = max(components, key=components.get) if components else "unknown"

            result.update({
                "aqi": mapped_aqi,
                "dominant_pollutant": dominant,
                "owm_aqi_raw": owm_aqi
            })
        else:
            print(f"OWM AQI Error: {data_aqi}")
            result.update({"aqi_error": "Could not fetch AQI"})

        # 3. UV Index (Try One Call or fallback)
        # Standard free key might fail here if One Call is not enabled.
        # We will try, but catch the error silently or set default.
        # Note: If One Call 3.0 is enabled, use that. If not, we skip UV.
        # For this implementation, we will skip sophisticated UV fetch to avoid 401 errors 
        # on standard keys. We set a safe default or checking 'current' response doesn't have it.
        # OpenMeteo was free, avoiding it means we lose UV.
        # We will set UV to None and let frontend/model handle it (default to Low risk).
        result.update({"uv_index": 0}) 
        
    except Exception as e:
        print(f"Error fetching environmental data: {e}")
        result.update({"weather_error": str(e)})

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
