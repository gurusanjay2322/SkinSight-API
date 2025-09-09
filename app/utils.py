import requests
import json
from .config import WAQI_API_KEY, OLLAMA_URL

# ----------------- Weather + AQI -----------------
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
def ask_llm(prompt, model="mistral"):
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)

        if response.status_code != 200:
            return {"error": response.text}

        data = response.json()
        raw_text = data.get("response", "").strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.strip('` \n')
            try:
                return json.loads(cleaned)
            except:
                return {"suggestions": [raw_text]}
    except Exception as e:
        return {"error": str(e)}
