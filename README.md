# SkinSight API

A Flask-based REST API that analyzes skin conditions from images and provides personalized skincare recommendations based on weather conditions and air quality data.

## Ìºü Features

- **Skin Type Classification**: Uses a pre-trained ResNet50 model to classify skin into 5 categories:
  - Acne
  - Burned
  - Dry
  - Normal
  - Oily

- **Weather Integration**: Fetches real-time weather data including:
  - Temperature (min/max)
  - UV Index
  - Air Quality Index (AQI)
  - Dominant pollutants

- **AI-Powered Recommendations**: Combines rule-based suggestions with AI-generated personalized advice using Ollama (Mistral model)

- **Risk Assessment**: Provides risk levels based on UV index and air quality

- **Interactive API Documentation**: Built-in Swagger UI for easy testing and documentation

## Ì∫Ä Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running locally
- WAQI API key for air quality data

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SkinSight_API
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   WAQI_API_KEY=your_waqi_api_key_here
   ```

5. **Download the pre-trained model**
   Ensure the model file `skin_type_classifier.pth` is in the `model/` directory.

6. **Start Ollama (for AI recommendations)**
   ```bash
   ollama serve
   ollama pull mistral
   ```

7. **Run the application**
   ```bash
   python run.py
   ```

The API will be available at `http://localhost:5000`

## Ì≥ö API Documentation

### Endpoints

#### POST `/predict`

Analyzes a skin image and provides personalized recommendations.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (file, required): Image of skin to classify
  - `lat` (number, required): Latitude of user location
  - `lon` (number, required): Longitude of user location

**Response:**
```json
{
  "predicted_class": "dry",
  "confidence": 0.85,
  "weather": {
    "temp_max": 25.5,
    "temp_min": 18.2,
    "uv_index": 7,
    "aqi": 120,
    "dominant_pollutant": "pm2.5",
    "city": "New York"
  },
  "risk_level": "Moderate",
  "rule_based_suggestions": [
    "Use sunscreen with SPF 30 or higher.",
    "Use a rich moisturizer daily."
  ],
  "genai_suggestions": [
    "Consider using a hydrating serum before moisturizer.",
    "Apply sunscreen every 2 hours when outdoors.",
    "Use gentle, fragrance-free products to avoid irritation."
  ]
}
```

### Interactive Documentation

Visit `http://localhost:5000/apidocs` to access the Swagger UI for interactive API testing.

## ÌøóÔ∏è Project Structure

```
SkinSight_API/
‚îú‚îÄ‚îÄ app/
‚îÇ   ‚îú‚îÄ‚îÄ __init__.py          # Flask app factory
‚îÇ   ‚îú‚îÄ‚îÄ routes.py            # API endpoints
‚îÇ   ‚îú‚îÄ‚îÄ model.py             # ML model and prediction logic
‚îÇ   ‚îú‚îÄ‚îÄ utils.py             # Weather and LLM utilities
‚îÇ   ‚îî‚îÄ‚îÄ config.py            # Configuration settings
‚îú‚îÄ‚îÄ model/
‚îÇ   ‚îî‚îÄ‚îÄ skin_type_classifier.pth  # Pre-trained model weights
‚îú‚îÄ‚îÄ myenv/                   # Virtual environment
‚îú‚îÄ‚îÄ run.py                   # Application entry point
‚îú‚îÄ‚îÄ requirements.txt         # Python dependencies
‚îú‚îÄ‚îÄ .env                     # Environment variables (not in repo)
‚îî‚îÄ‚îÄ README.md               # This file
```

## Ì¥ß Configuration

### Environment Variables

- `WAQI_API_KEY`: API key for World Air Quality Index service
- `OLLAMA_URL`: URL for Ollama API (default: http://localhost:11434/api/generate)

### Model Configuration

The application uses a ResNet50 model fine-tuned for skin classification. The model expects:
- Input size: 224x224 pixels
- Color channels: RGB
- Normalization: ImageNet standards

## Ìºê External Services

### Weather Data
- **Open-Meteo API**: Free weather data service
- Provides temperature, UV index, and timezone information

### Air Quality
- **WAQI API**: World Air Quality Index service
- Requires API key registration at [waqi.info](https://waqi.info)

### AI Recommendations
- **Ollama**: Local LLM server
- Uses Mistral model for generating personalized skincare advice

## Ì∑™ Testing

You can test the API using the Swagger UI at `http://localhost:5000/apidocs` or with curl:

```bash
curl -X POST "http://localhost:5000/predict" \
  -F "image=@path/to/your/image.jpg" \
  -F "lat=40.7128" \
  -F "lon=-74.0060"
```

## Ìª†Ô∏è Development

### Adding New Skin Types

1. Retrain the model with new classes
2. Update the `class_names` list in `app/model.py`
3. Modify the rule-based suggestions logic

### Extending Weather Data

Add new weather parameters in `app/utils.py` by modifying the Open-Meteo API request URL.

### Customizing AI Recommendations

Modify the prompt template in `app/model.py` to change how the LLM generates suggestions.

## Ì≥ù License

This project is licensed under the MIT License - see the LICENSE file for details.

## Ì¥ù Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Ì≥û Support

For questions or issues, please open an issue on the GitHub repository.

## Ì¥Æ Future Enhancements

- [ ] Support for multiple image formats
- [ ] Batch processing capabilities
- [ ] User authentication and history
- [ ] Mobile app integration
- [ ] Additional skin condition classifications
- [ ] Integration with more weather services
- [ ] Real-time notifications for high-risk conditions
