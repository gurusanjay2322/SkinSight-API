from flask import Flask, request, jsonify
import requests
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from dotenv import load_dotenv
import subprocess
import json

# Load environment variables
load_dotenv()
WAQI_API_KEY = os.getenv("WAQI_API_KEY")

app = Flask(__name__)

# ----------------- Load the model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)

model.load_state_dict(torch.load("model/skin_type_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------- Transform pipeline -----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['acne', 'burned', 'dry', 'normal', 'oily']

# ----------------- Weather and AQI retrieval -----------------
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
        url_aqi = (
            f"https://api.waqi.info/feed/geo:{lat};{lon}/"
            f"?token={WAQI_API_KEY}"
        )
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
import requests

import json
import requests

def ask_llm(prompt, model="mistral"):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code != 200:
            return {"error": response.text}

        data = response.json()
        raw_text = data.get("response", "").strip()

        # Try to parse LLM response as JSON
        try:
            parsed = json.loads(raw_text)
            return parsed
        except json.JSONDecodeError:
            # If the model wrapped JSON in quotes, fix it
            cleaned = raw_text.strip('` \n')
            try:
                return json.loads(cleaned)
            except:
                # fallback to wrapping
                return {"suggestions": [raw_text]}
    except Exception as e:
        return {"error": str(e)}




# ----------------- /predict endpoint -----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_file = request.files['image']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    if not lat or not lon:
        return jsonify({"error": "Please provide lat and lon"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude"}), 400

    # Process the uploaded image
    img_bytes = image_file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()

    # Get weather and AQI info
    weather_data = get_weather(lat, lon)

    # Rule-based risk + suggestions
    risk_level = "Low"
    suggestions = []
    uv_index = weather_data.get("uv_index")
    if uv_index is not None:
        if uv_index >= 8:
            risk_level = "High"
            suggestions.append("Avoid direct sunlight between 10 AM and 4 PM.")
        elif uv_index >= 6:
            risk_level = "Moderate"
            suggestions.append("Use sunscreen with SPF 30 or higher.")

    aqi = weather_data.get("aqi")
    if aqi is not None:
        if aqi > 200:
            risk_level = "Very High"
            suggestions.append("Avoid outdoor activities and wear a pollution mask.")
        elif aqi > 150:
            if risk_level != "Very High":
                risk_level = "High"
            suggestions.append("Limit prolonged or heavy exertion outdoors.")
        elif aqi > 100:
            if risk_level == "Low":
                risk_level = "Moderate"
            suggestions.append("Consider reducing outdoor activities.")

    # Skin type rules
    if predicted_class == "dry":
        suggestions.append("Use a rich moisturizer daily.")
    if predicted_class == "oily":
        suggestions.append("Use non-comedogenic products.")
    if predicted_class == "acne":
        suggestions.append("Use products with salicylic acid.")
    if predicted_class == "burned":
        suggestions.append("Apply soothing creams and avoid sun.")
    if predicted_class == "normal":
        suggestions.append("Maintain a balanced skincare routine.")

    # ----------- Call LLM for extra personalized advice -----------
    llm_prompt = f"""
    You are a dermatologist assistant.
    The user’s detected skin type is: {predicted_class}.
    Confidence: {round(confidence,2)}.
    Weather conditions: {weather_data}.
    Rule-based risk level: {risk_level}.
    Rule-based suggestions: {suggestions}.

    Respond ONLY in valid JSON.
    Format:
    {{
    "suggestions": [
        "First tip...",
        "Second tip...",
        "Third tip..."
    ]
    }}
    """
    llm_response = ask_llm(llm_prompt)

    llm_response = ask_llm(llm_prompt, model="mistral")

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "weather": weather_data,
        "risk_level": risk_level,
        "rule_based_suggestions": suggestions,
        "genai_suggestions": llm_response.get("suggestions", [])
    })



if __name__ == "__main__":
    app.run(debug=True)
