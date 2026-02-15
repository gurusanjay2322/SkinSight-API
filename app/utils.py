import requests
import json
import time
from functools import wraps
from .config import OPENWEATHERMAP_API_KEY, OLLAMA_URL, OLLAMA_API_KEY


# ----------------- Retry Decorator with Exponential Backoff -----------------
def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.RequestException,
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        print(
                            f"⚠️ Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"❌ All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            raise last_exception

        return wrapper

    return decorator


@retry_with_backoff(max_retries=2, base_delay=0.5)
def _fetch_url(url, timeout=10):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _fetch_url_safe(url, timeout=10, default=None):
    try:
        return _fetch_url(url, timeout)
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return default


# ----------------- WEATHER -----------------
def get_weather(lat, lon):
    result = {}

    # --- Forecast + UV ---
    try:
        url_forecast = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,uv_index_max,sunrise,sunset&timezone=auto&forecast_days=3"
        data_forecast = _fetch_url_safe(url_forecast, timeout=10)

        if data_forecast and "daily" in data_forecast:
            daily = data_forecast["daily"]
            result.update(
                {
                    "forecast": {
                        "dates": daily.get("time", []),
                        "temp_max": daily.get("temperature_2m_max", []),
                        "temp_min": daily.get("temperature_2m_min", []),
                        "uv_max": daily.get("uv_index_max", []),
                        "weather_code": daily.get("weather_code", []),
                        "sunrise": daily.get("sunrise", []),
                        "sunset": daily.get("sunset", []),
                    },
                    "uv_index": round(float(daily.get("uv_index_max", [0])[0]), 1),
                }
            )
            print("✅ Forecast/UV data fetched")
    except Exception as e:
        print(f"⚠️ Forecast fetch failed: {e}")

    if not OPENWEATHERMAP_API_KEY:
        result.update({"weather_error": "Missing OPENWEATHERMAP_API_KEY"})
        result.setdefault("uv_index", 0)
        return result

    # --- Current Weather ---
    try:
        url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        data_weather = _fetch_url_safe(url_weather, timeout=10, default={})

        if data_weather and "main" in data_weather:
            main = data_weather.get("main", {})
            sys = data_weather.get("sys", {})
            weather_info = data_weather.get("weather", [{}])[0]

            result.update(
                {
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
                    "sunrise": sys.get("sunrise"),
                    "sunset": sys.get("sunset"),
                    "wind_speed": data_weather.get("wind", {}).get("speed", 0),
                    "clouds": data_weather.get("clouds", {}).get("all", 0),
                }
            )
            print(f"✅ Weather data fetched for {result.get('city')}")
        else:
            result.update({"weather_error": "Failed to fetch weather data"})
    except Exception as e:
        print(f"❌ Exception fetching weather: {e}")
        result.update({"weather_error": str(e)})

    # --- AQI ---
    try:
        url_aqi = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
        data_aqi = _fetch_url_safe(url_aqi, timeout=10, default={})

        if data_aqi and data_aqi.get("list"):
            owm_aqi = data_aqi["list"][0]["main"]["aqi"]
            components = data_aqi["list"][0].get("components", {})

            aqi_map = {1: 20, 2: 80, 3: 120, 4: 180, 5: 250}
            dominant = max(components, key=components.get) if components else "unknown"

            result.update(
                {
                    "aqi": aqi_map.get(owm_aqi, 100),
                    "dominant_pollutant": dominant,
                    "owm_aqi_raw": owm_aqi,
                }
            )
            print(f"✅ AQI fetched")
        else:
            result.update({"aqi": 0})
    except Exception as e:
        print(f"❌ AQI error: {e}")
        result.update({"aqi": 0})

    result.setdefault("uv_index", 0)
    return result


# ----------------- AI Suggestions -----------------
MODEL_PRIORITY = [
    "gpt-4o",  # 🔥 Extremely stable structured output (primary)
    "gpt-4o-mini",  # ⚡ Fast fallback
    "gpt-3.5-turbo",  # 🛟 Legacy fallback
]


def ask_llm(prompt: str):
    import os
    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        print("❌ Missing OPENAI_API_KEY")
        return {"error": "Missing API key", "suggestions": []}

    for model in MODEL_PRIORITY:
        try:
            print(f"🤖 Calling LLM (model: {model})...")

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """
You are GlowBot's suggestion engine.

Output valid JSON.
If the user provides a specific JSON schema, FOLLOW IT EXACTLY.
Otherwise return:
{ "suggestions": ["tip1", "tip2", "tip3"] }

Rules:
- OTC skincare only
- No prescriptions
- No markdown
""",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()

            if not raw_text:
                continue

            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "suggestions" in parsed:
                return parsed

        except openai.RateLimitError:
            print(f"🛑 Rate limit hit for {model}, falling back...")
            continue
        except Exception as e:
            print(f"❌ Exception (model {model}): {e}")
            if "model_not_found" in str(e).lower():
                continue
            return {"error": str(e), "suggestions": []}

    return {"error": "All models exhausted", "suggestions": []}
