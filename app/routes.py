from flask import Blueprint, request, jsonify
from .utils import get_weather, ask_llm
from .model import predict_skin

bp = Blueprint("api", __name__)

@bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict Skin Type from Image + Location
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Image of skin to classify
      - name: lat
        in: formData
        type: number
        required: true
        description: Latitude of user
      - name: lon
        in: formData
        type: number
        required: true
        description: Longitude of user
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            predicted_class:
              type: string
            confidence:
              type: number
            weather:
              type: object
            risk_level:
              type: string
            rule_based_suggestions:
              type: array
              items:
                type: string
            genai_suggestions:
              type: array
              items:
                type: string
    """
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

    weather_data = get_weather(lat, lon)
    result = predict_skin(image_file, lat, lon, weather_data, ask_llm)

    return jsonify(result)
