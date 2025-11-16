from flask import Blueprint, request, jsonify
from .utils import get_weather, ask_llm
from .model import predict_skin
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import requests
from io import BytesIO

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

def detect_disease(image_file):
    """
    Call the disease detection API on port 5001 to detect skin diseases.
    Returns disease prediction result or None if API call fails.
    """
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Read the file content
        image_bytes = image_file.read()
        image_file.seek(0)  # Reset again for potential reuse
        
        # Prepare the image file for forwarding
        files = {
            'image': (
                image_file.filename or 'image.jpg',
                BytesIO(image_bytes),
                image_file.content_type or 'image/jpeg'
            )
        }
        
        # Call disease detection API
        disease_api_url = "http://localhost:5001/predict"
        response = requests.post(
            disease_api_url,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            disease_data = response.json()
            print(f"[DiseaseDetection] Success: {disease_data}")
            return disease_data
        else:
            print(f"[DiseaseDetection] API returned status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[DiseaseDetection] Error calling disease API: {str(e)}")
        return None
    except Exception as e:
        print(f"[DiseaseDetection] Unexpected error: {str(e)}")
        return None

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
    
    # Call disease detection API
    image_file.seek(0)  # Reset file pointer
    disease_result = detect_disease(image_file)
    
    # Predict skin type with disease information
    image_file.seek(0)  # Reset file pointer again
    result = predict_skin(image_file, lat, lon, weather_data, ask_llm, disease_result)

    return jsonify(result)
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
        
        # If skin is valid, also call disease detection API
        disease_result = None
        if valid:
            try:
                image_file.seek(0)  # Reset file pointer
                disease_result = detect_disease(image_file)
            except Exception as e:
                print(f"[validSkin] Disease detection failed: {str(e)}")
                # Continue even if disease detection fails

        response = {
            "valid": valid,
            "human_ratio": float(human_ratio),
            "skin_ratio": float(skin_ratio),
            "message": "Skin detected" if valid else "No visible skin detected"
        }
        
        # Include disease result if available
        if disease_result:
            response["disease"] = disease_result

        return jsonify(response)
    except Exception as e:
        print("Error during validSkin:", e)
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500