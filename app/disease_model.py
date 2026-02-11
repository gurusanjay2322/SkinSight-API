import tensorflow as tf
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import os
import h5py
import json

# HAM10000 class names (7 classes)
CLASS_NAMES = [
    'Actinic Keratoses (AKIEC)',
    'Basal Cell Carcinoma (BCC)',
    'Benign Keratosis (BKL)',
    'Dermatofibroma (DF)',
    'Melanoma (MEL)',
    'Melanocytic Nevi (NV)',
    'Vascular Lesion (VASC)'
]

CLASS_CODES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    'akiec': {'name': 'Actinic Keratosis', 'severity': 'Medium', 'description': 'Pre-cancerous rough patch caused by sun damage.'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'severity': 'High', 'description': 'Common skin cancer. Slow-growing but requires medical attention.'},
    'bkl': {'name': 'Benign Keratosis', 'severity': 'Low', 'description': 'Non-cancerous skin growth. Usually harmless.'},
    'df': {'name': 'Dermatofibroma', 'severity': 'Low', 'description': 'Benign fibrous nodule. Usually harmless.'},
    'mel': {'name': 'Melanoma', 'severity': 'Critical', 'description': 'Serious skin cancer - consult a dermatologist immediately.'},
    'nv': {'name': 'Melanocytic Nevus', 'severity': 'Low', 'description': 'Common mole. Usually benign.'},
    'vasc': {'name': 'Vascular Lesion', 'severity': 'Low', 'description': 'Blood vessel-related marking. Usually benign.'}
}

model = None
model_load_error = None
TEMPERATURE = 2.77

def fix_model_config(model_path):
    """
    Fix model config by removing incompatible parameters.
    Returns path to fixed model or None if fix not needed/possible.
    """
    try:
        fixed_path = model_path.replace('.h5', '_fixed.h5')
        
        # If already fixed, return the fixed path
        if os.path.exists(fixed_path):
            return fixed_path
            
        print("  Fixing model compatibility...")
        
        # Copy and fix the h5 file
        import shutil
        shutil.copy(model_path, fixed_path)
        
        with h5py.File(fixed_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Recursively fix layer configs
                def fix_layer_config(layer_config):
                    if isinstance(layer_config, dict):
                        # Remove 'groups' from DepthwiseConv2D
                        if layer_config.get('class_name') == 'DepthwiseConv2D':
                            if 'config' in layer_config and 'groups' in layer_config['config']:
                                del layer_config['config']['groups']
                        
                        # Recurse into nested configs
                        for key, value in layer_config.items():
                            if isinstance(value, (dict, list)):
                                fix_layer_config(value)
                    elif isinstance(layer_config, list):
                        for item in layer_config:
                            fix_layer_config(item)
                
                fix_layer_config(config)
                
                # Save fixed config
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                
        print("  ✅ Model compatibility fixed!")
        return fixed_path
        
    except Exception as e:
        print(f"  ⚠️ Could not fix model: {e}")
        return None

def load_model():
    """Load the EfficientNetV2S skin cancer classifier from HuggingFace."""
    global model, model_load_error
    
    if model is not None:
        return model
        
    model_load_error = None
    print("Loading skin disease detection model from HuggingFace...")
    print("  Model: Miguel764/efficientnetv2s-skin-cancer-classifier")
    print("  Dataset: HAM10000 (7 classes)")
    print("  Accuracy: 88%")
    
    try:
        # Download the .h5 model file from HuggingFace
        model_path = hf_hub_download(
            repo_id="Miguel764/efficientnetv2s-skin-cancer-classifier",
            filename="efficientnetv2s.h5"
        )
        print(f"  Downloaded to: {model_path}")
        
        # Try to load directly first
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
            if 'groups' in str(e):
                # Fix compatibility issue and retry
                fixed_path = fix_model_config(model_path)
                if fixed_path:
                    model = tf.keras.models.load_model(fixed_path, compile=False)
                else:
                    raise
            else:
                raise
        
        print("✅ Disease detection model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ Error loading disease model: {error_msg}")
        print(traceback.format_exc())
        model_load_error = error_msg
        model = None
        
    return model

def get_model():
    return model

def get_model_error():
    return model_load_error

def preprocess_image(image, img_size=224):
    img = image.convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32) / 255.0
    return img_array

def apply_temperature_scaling(logits, temperature=TEMPERATURE):
    scaled_logits = logits / temperature
    return tf.nn.softmax(scaled_logits).numpy()

def predict(image):
    """Runs inference on a single image."""
    global model
    
    if model is None:
        load_model()
        if model is None:
            raise Exception(f"Model not loaded. Error: {model_load_error}")
    
    processed_image = preprocess_image(image)
    raw_output = model.predict(processed_image, verbose=0)
    predictions = apply_temperature_scaling(raw_output)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    class_code = CLASS_CODES[predicted_class_index]
    class_info = CLASS_INFO[class_code]
    predicted_class_name = class_info['name']
    confidence = float(predictions[0][predicted_class_index])
    
    print(f"[DiseaseModel] Prediction: {predicted_class_name} ({confidence:.2%})")
    
    return predicted_class_name, confidence

def get_class_info(class_name):
    for code, info in CLASS_INFO.items():
        if info['name'] == class_name:
            return info
    return {'name': class_name, 'severity': 'Unknown', 'description': 'Skin condition detected'}
