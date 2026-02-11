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

def predict_skin(image_file, lat, lon, weather_data, llm_func, disease_result=None):
    from datetime import datetime, timezone, timedelta

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
        
        # Get all class probabilities for the confidence chart
        all_probabilities = probabilities[0].cpu().numpy()
        class_scores = {class_names[i]: round(float(all_probabilities[i]) * 100, 1) for i in range(len(class_names))}

    # ---- Time-of-day awareness ----
    tz_offset = weather_data.get("timezone", 19800)  # Default IST offset in seconds
    if isinstance(tz_offset, int):
        user_tz = timezone(timedelta(seconds=tz_offset))
    else:
        user_tz = timezone(timedelta(hours=5, minutes=30))  # Fallback IST
    
    now = datetime.now(user_tz)
    current_hour = now.hour
    current_time_str = now.strftime("%I:%M %p")
    
    # Determine time period
    if 5 <= current_hour < 9:
        time_period = "early_morning"
        time_label = "Early Morning"
    elif 9 <= current_hour < 12:
        time_period = "morning"
        time_label = "Morning"
    elif 12 <= current_hour < 16:
        time_period = "afternoon"
        time_label = "Afternoon"
    elif 16 <= current_hour < 19:
        time_period = "evening"
        time_label = "Evening"
    else:
        time_period = "night"
        time_label = "Night"

    # Rule-based risk + suggestions
    risk_level = "Low"
    suggestions = []

    uv_index = weather_data.get("uv_index")
    if uv_index is not None:
        if uv_index >= 8:
            risk_level = "High"
            if time_period in ("morning", "afternoon"):
                suggestions.append(f"⚠️ UV is VERY HIGH right now ({current_time_str}). Stay indoors or use SPF 50+.")
            else:
                suggestions.append("UV is extreme today. Apply SPF 50+ if you go out tomorrow morning.")
        elif uv_index >= 6:
            risk_level = "Moderate"
            if time_period in ("morning", "afternoon"):
                suggestions.append(f"UV is elevated right now ({current_time_str}). Apply sunscreen with SPF 30+.")
            else:
                suggestions.append("Use sunscreen with SPF 30 or higher during daytime hours.")
        elif uv_index >= 3:
            suggestions.append("Moderate UV today. Basic sun protection recommended during peak hours (10 AM - 4 PM).")

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

    # Time-specific skincare tips
    if time_period == "night":
        suggestions.append("🌙 It's nighttime — perfect for your evening skincare routine. Apply night cream/serum now.")
    elif time_period == "early_morning":
        suggestions.append("🌅 Start your day with a gentle cleanser and moisturizer before heading out.")
    elif time_period == "evening":
        suggestions.append("🌆 Sun is setting. Great time for outdoor activities with minimal UV risk.")

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
    
    # Enhance suggestions based on disease detection
    disease_info = None
    if disease_result:
        disease_class = disease_result.get("predicted_class", "")
        disease_confidence = disease_result.get("confidence", 0)
        disease_info = {
            "predicted_class": disease_class,
            "confidence": disease_confidence,
            "confidence_percentage": disease_result.get("confidence_percentage", f"{disease_confidence:.2%}")
        }
        
        # Add disease-specific suggestions
        if disease_confidence > 0.5:  # Only if confidence is reasonable
            if "Carcinoma" in disease_class or "Squamous" in disease_class or "Basal" in disease_class:
                risk_level = "Very High"
                suggestions.insert(0, f"⚠️ Detected: {disease_class}. Please consult a dermatologist immediately.")
                suggestions.insert(1, "Avoid sun exposure and use high SPF sunscreen.")
            elif "Keratosis" in disease_class:
                if risk_level == "Low":
                    risk_level = "Moderate"
                suggestions.insert(0, f"Detected: {disease_class}. Consider regular skin monitoring.")
                suggestions.insert(1, "Use sunscreen and avoid excessive sun exposure.")
            elif "Nevus" in disease_class:
                suggestions.insert(0, "Monitor any changes in size, shape, or color.")
                suggestions.insert(1, "Regular dermatological check-ups recommended.")
            elif "Lesion" in disease_class:
                suggestions.insert(0, "Monitor the lesion for any changes.")
                suggestions.insert(1, "Consider consulting a dermatologist for evaluation.")

    # ----------- Call LLM for extra personalized advice ----------- 
    disease_context = ""
    if disease_info:
        disease_context = f"\nDetected skin condition: {disease_info['predicted_class']} (confidence: {disease_info['confidence_percentage']})."
    
    llm_prompt = f"""
    You are a dermatologist assistant.
    The user's detected skin type is: {predicted_class}.
    Confidence: {round(confidence,2)}.
    Weather conditions: {weather_data}.
    Current local time: {current_time_str} ({time_label}).
    Rule-based risk level: {risk_level}.
    Rule-based suggestions: {suggestions}.{disease_context}

    Give 3-5 personalized skincare tips that are RELEVANT TO THE CURRENT TIME OF DAY.
    For example, if it's morning, focus on morning routine. If night, focus on evening routine.
    
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
    llm_response = llm_func(llm_prompt)

    # Add time context to weather data
    weather_data["current_time"] = current_time_str
    weather_data["time_period"] = time_label

    result = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "class_scores": class_scores,  # All 5 skin type probabilities
        "weather": weather_data,
        "risk_level": risk_level,
        "rule_based_suggestions": suggestions,
        "genai_suggestions": llm_response.get("suggestions", [])
    }
    
    # Include disease information if available
    if disease_info:
        result["disease"] = disease_info

    return result

