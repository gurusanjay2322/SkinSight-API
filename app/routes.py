from flask import Blueprint, request, jsonify
from PIL import Image
from io import BytesIO
from .utils import get_weather, ask_llm
from .model import predict_skin
from .disease_model import predict as predict_disease
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import requests
from io import BytesIO

bp = Blueprint("api", __name__)

# Initialize OpenAI
import os
import openai
from flask import Response, stream_with_context

openai.api_key = os.getenv("OPENAI_API_KEY")

@bp.route("/chat", methods=["POST"])
def chat_with_context():
    """
    Chat with context about skin analysis
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            context:
              type: object
              description: The full skin analysis result
            question:
              type: string
              description: User's question
            history:
              type: array
              items:
                type: object
                properties:
                  role:
                    type: string
                  content:
                    type: string
    responses:
      200:
        description: Chat response
    """
    data = request.json
    context = data.get('context', {})
    question = data.get('question', '')
    history = data.get('history', [])

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # System prompt engineering
    system_prompt = f"""You are 'GlowBot', an expert AI Demotologist assistant. 
    You have analyzed the user's skin and here are the results:
    
    Condition: {context.get('predictedClass', 'Unknown')}
    Confidence: {context.get('confidence', 0)}
    Risk Level: {context.get('riskLevel', 'Unknown')}
    Weather Context: UV Index {context.get('weather', {}).get('uv_index', 'N/A')}, Humidity {context.get('weather', {}).get('humidity', 'N/A')}%
    
    Your goal is to answer the user's questions specifically about THEIR skin condition based on this data.
    - Be empathetic but professional.
    - Do NOT give medical prescriptions, only OTC advice and routine tips.
    - If the risk is High/Very High, strongly advise seeing a doctor.
    - Keep answers concise (under 3 sentences) unless asked for details.
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (last 5 messages to save tokens)
    messages.extend(history[-5:])
    
    # Add current question
    messages.append({"role": "user", "content": question})

    def generate():
        try:
            # Use OpenAI ChatCompletion (Streaming)
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"

    return Response(stream_with_context(generate()), mimetype='text/plain')


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

def detect_disease_local(image_file):
    """
    Detect skin disease using the local loaded model.
    """
    from .disease_model import get_model, get_model_error, predict as predict_disease_func
    
    try:
        # Check if model is loaded
        model = get_model()
        if model is None:
            error = get_model_error()
            print(f"[DiseaseDetection] Model not loaded. Error: {error}")
            return None
        
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Read the file content
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        
        print(f"[DiseaseDetection] Running prediction...")
        
        # Predict
        predicted_class, confidence = predict_disease_func(image)
        
        disease_data = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "confidence_percentage": f"{confidence:.2%}"
        }
        
        print(f"[DiseaseDetection] Success: {disease_data}")
        return disease_data

    except Exception as e:
        import traceback
        print(f"[DiseaseDetection] Error: {str(e)}")
        print(traceback.format_exc())
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
            disease:
              type: object
              properties:
                predicted_class:
                  type: string
                confidence:
                  type: number
                confidence_percentage:
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
    
    # Call disease detection locally
    image_file.seek(0)  # Reset file pointer
    disease_result = detect_disease_local(image_file)
    
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
                disease_result = detect_disease_local(image_file)
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