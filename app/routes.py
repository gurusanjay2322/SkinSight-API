from flask import Blueprint, request, jsonify
from .utils import get_weather, ask_llm
from .model import predict_skin
import cv2
import numpy as np
import mediapipe as mp
import tempfile

bp = Blueprint("api", __name__)
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def detect_skin_in_image(file_storage, human_threshold=0.1, skin_threshold=0.25):
    """
    Improved skin detection: requires both human-like segmentation and realistic skin tone clustering.
    Rejects false positives (like objects, cans, or backgrounds).
    """
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file_storage.save(tmp.name)
        path = tmp.name

    img = cv2.imread(path)
    if img is None:
        return False, 0, 0

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: MediaPipe segmentation
    seg = mp_selfie.process(rgb)
    mask = seg.segmentation_mask
    if mask is None:
        return False, 0, 0

    human_ratio = float(np.mean(mask > 0.5))

    # Step 2: Face detection (bonus validation)
    faces = mp_face.process(rgb)
    has_face = bool(faces.detections)

    # Step 3: Stricter HSV range (avoid object false positives)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([17, 200, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower, upper)

    # Apply the human mask (only count skin-colored pixels inside segmented human regions)
    skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=(mask > 0.5).astype(np.uint8) * 255)
    skin_ratio = float(np.count_nonzero(skin_mask) / (img.size / 3))

    # Step 4: Add texture sanity check — skin has mild color variation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    texture_ok = variance > 100  # filter out flat or glossy surfaces (like cans/paper)

    # Final validity rule
    valid = ((human_ratio > human_threshold and skin_ratio > skin_threshold and texture_ok) or has_face)

    print(f"[validSkin] human_ratio={human_ratio:.3f}, skin_ratio={skin_ratio:.3f}, variance={variance:.2f}, face={has_face}")

    return valid, human_ratio, skin_ratio

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
@bp.route("/validSkin", methods=["POST"])
@bp.route("/validSkin", methods=["POST"])
def valid_skin():
    """
    Validate if uploaded image likely contains visible skin.
    Expects multipart/form-data with 'image' field.
    Returns JSON: { valid: bool, human_ratio: float, skin_ratio: float }
    """
    if 'image' not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_file = request.files['image']

    try:
        valid, human_ratio, skin_ratio = detect_skin_in_image(image_file)

        # ⚙️ Cast NumPy bool to Python bool
        valid = bool(valid)

        return jsonify({
            "valid": valid,
            "human_ratio": float(human_ratio),
            "skin_ratio": float(skin_ratio),
            "message": "Skin detected" if valid else "No visible skin detected"
        })
    except Exception as e:
        print("Error during validSkin:", e)
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    """
    Validate if uploaded image likely contains visible skin.
    Expects multipart/form-data with 'image' field.
    Returns JSON: { valid: bool, human_ratio: float, skin_ratio: float }
    """
    if 'image' not in request.files:
        return jsonify({"error": "Image file is missing"}), 400

    image_file = request.files['image']

    try:
        valid, human_ratio, skin_ratio = detect_skin_in_image(image_file)
        return jsonify({
            "valid": valid,
            "human_ratio": round(human_ratio, 3),
            "skin_ratio": round(skin_ratio, 3),
            "message": "Skin detected" if valid else "No visible skin detected"
        })
    except Exception as e:
        print("Error during validSkin:", e)
        return jsonify({"error": "Failed to process image"}), 500