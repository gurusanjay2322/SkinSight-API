from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
WAQI_API_KEY = os.getenv("WAQI_API_KEY")

app = Flask(__name__)

@app.route('/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', '').strip()
    lon = request.args.get('lon', '').strip()

    if not lat or not lon:
        return jsonify({"error": "Please provide lat and lon query parameters."}), 400

    result = {}

    # ----- Open-Meteo for temp and UV -----
    try:
        url_meteo = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,uv_index_max"
            f"&timezone=auto"
        )
        response = requests.get(url_meteo)
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

    # ----- WAQI for AQI -----
    try:
        url_aqi = (
            f"https://api.waqi.info/feed/geo:{lat};{lon}/"
            f"?token={WAQI_API_KEY}"
        )
        response = requests.get(url_aqi)
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

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
