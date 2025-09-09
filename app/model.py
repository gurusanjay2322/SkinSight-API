import torch
from torchvision import models, transforms
from PIL import Image
import io

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Model -----------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load("model/skin_type_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------- Classes -----------------
class_names = ['acne', 'burned', 'dry', 'normal', 'oily']

# ----------------- Transform -----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_skin(image_file, lat, lon, weather_data, llm_func):
    # Preprocess image
    img_bytes = image_file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()

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
    llm_response = llm_func(llm_prompt, model="mistral")

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "weather": weather_data,
        "risk_level": risk_level,
        "rule_based_suggestions": suggestions,
        "genai_suggestions": llm_response.get("suggestions", [])
    }
