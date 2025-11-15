import requests
import json
from .config import WAQI_API_KEY, OLLAMA_URL, OLLAMA_API_KEY
from dotenv import load_dotenv
# ----------------- Weather + AQI -----------------
load_dotenv() 
def get_weather(lat, lon):
    result = {}
    try:
        url_meteo = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,uv_index_max"
            f"&timezone=auto"
        )
        response = requests.get(url_meteo, timeout=10)
        data = response.json()

        daily = data.get("daily", {})
        result.update({
            "temp_max": daily.get("temperature_2m_max", [None])[0],
            "temp_min": daily.get("temperature_2m_min", [None])[0],
            "uv_index": daily.get("uv_index_max", [None])[0],
            "timezone": data.get("timezone")
        })
    except Exception as e:
        result.update({"weather_error": str(e)})

    try:
        url_aqi = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_API_KEY}"
        response = requests.get(url_aqi, timeout=10)
        data = response.json()

        if data.get("status") == "ok":
            aqi_data = data.get("data", {})
            result.update({
                "aqi": aqi_data.get("aqi"),
                "dominant_pollutant": aqi_data.get("dominentpol"),
                "city": aqi_data.get("city", {}).get("name")
            })
        else:
            result.update({"aqi_error": data.get("data")})
    except Exception as e:
        result.update({"aqi_error": str(e)})

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
