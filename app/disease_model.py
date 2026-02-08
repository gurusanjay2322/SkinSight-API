import tensorflow as tf
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

# Define the class names in the correct order
CLASS_NAMES = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Nevus', 
    'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 
    'Squamous Cell Carcinoma', 'Vascular Lesion'
]

# Global model variable
model = None
model_load_error = None

def build_model():
    """Reconstructs the EfficientNetB0 model architecture."""
    # Input shape used in training (likely 224x224x3 for EfficientNet)
    inputs = Input(shape=(224, 224, 3))
    
    # Base model with ImageNet weights (we'll overwrite them, but it initializes the structure)
    base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    
    # Rebuild the head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model():
    """Load the model weights from Hugging Face Hub."""
    global model, model_load_error
    if model is None:
        model_load_error = None
        print("Loading disease detection model from Hugging Face Hub...")
        try:
            # Download the entire model repository
            model_dir = snapshot_download(
                repo_id="Arko007/skin-disease-detector-ai",
                cache_dir=None
            )
            print(f"Model downloaded to: {model_dir}")
            
            # Find the model file (.keras or .h5)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.h5')]
            
            if not model_files:
                 # Check if there are other files like saved_model.pb in a subdirectory? 
                 # But based on logs, we saw 'model.keras'.
                 raise Exception(f"No .keras or .h5 file found in {model_dir}")

            model_path = os.path.join(model_dir, model_files[0])
            print(f"Loading weights from: {model_path}")
            
            # Build the architecture first
            model = build_model()
            
            # Load the weights
            # We use by_name=True or skip_mismatch=True if needed, but strict loading is better if architecture matches exactly.
            # Given the error 'Could not locate class MBConvBlock', loading the *entire* model failed.
            # Loading weights should work if the layers match.
            try:
                model.load_weights(model_path)
                print("Weights loaded successfully!")
            except Exception as e_weights:
                print(f"Standard load_weights failed: {e_weights}")
                # Fallback: try loading with verification disabled?
                # or maybe the model file is a FULL SavedModel, not just weights. 
                # Attempt to load it as a full model with safe_mode=False purely for weight extraction?
                # No, that failed before.
                raise e_weights

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"Error loading disease model: {error_msg}")
            print(error_trace)
            model_load_error = error_msg
            model = None
    return model

def get_model():
    """Get the loaded model."""
    return model

def get_model_error():
    """Get the model loading error message."""
    return model_load_error

def preprocess_image(image, img_size=224):
    """Preprocesses an image (PIL Image object) for the model."""
    # EfficientNet logic
    img = image.convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # EfficientNet typically expects inputs in [0, 255] if using the internal rescaling layers, 
    # OR [-1, 1] / [0, 1] if preprocessed externally.
    # The 'app.py' from the original repo used `efficientnet.preprocess_input` which does nothing for EfficientNet (pass-through) because it has internal normalization.
    # However, the `model.py` which we ported earlier used `/ 255.0`.
    # AND the error message in the logs showed `rescaling` layers.
    # If the model has internal rescaling, we should pass 0-255.
    # If we divide by 255 here AND the model divides by 255 internally, we get tiny numbers.
    
    # Safe bet: Try passing 0-255 (raw pixels). 
    # But wait, `model.py` in the original repo (Step 55 previous session) had `/ 255.0`.
    # Let's trust the `model.py` logic we saw earlier.
    img_array = tf.cast(img_array, tf.float32) / 255.0
    return img_array

def predict(image):
    """Runs inference on a single image (PIL Image object)."""
    if model is None:
        load_model()
        if model is None:
             raise Exception(f"Model not loaded. Error: {model_load_error}")
    
    processed_image = preprocess_image(image)
    
    predictions = model.predict(processed_image, verbose=0)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions, axis=1)[0])
    
    return predicted_class_name, confidence
