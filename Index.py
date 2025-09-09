from flask import Flask, request, jsonify
import requests
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
WAQI_API_KEY = os.getenv("WAQI_API_KEY")

app = Flask(__name__)

# Load the model once when the app starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load("model/skin_type_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['acne', 'burned', 'dry', 'normal', 'oily']

@app.route('/weather', methods=['GET'])
def get_weather(lat, lon):
    result = {}
    # Open-Meteo
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

    # WAQI
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

    return result


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

    # Process image
    img_bytes = image_file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()

    # Get weather and AQI data
    weather_data = get_weather(lat, lon)

    # Determine risk level and suggestions based on skin type and AQI/UV
    risk_level = "Low"
    suggestions = []

    if weather_data.get("uv_index") and weather_data["uv_index"] > 7:
        risk_level = "High"
        suggestions.append("Avoid direct sunlight between 10 AM - 4 PM.")
    if weather_data.get("aqi") and weather_data["aqi"] > 150:
        risk_level = "High"
        suggestions.append("Limit outdoor activities and wear a mask.")
    if predicted_class == "dry":
        suggestions.append("Use moisturizer regularly.")
    if predicted_class == "oily":
        suggestions.append("Avoid oily skincare products.")

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "weather": weather_data,
        "risk_level": risk_level,
        "suggestions": suggestions
    })


if __name__ == "__main__":
    app.run(debug=True)
