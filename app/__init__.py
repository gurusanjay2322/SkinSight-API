from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

def create_app():
    app = Flask(__name__)

    # Swagger configuration
    app.config['SWAGGER'] = {
        'title': 'Skin Analyzer API',
        'uiversion': 3
    }

    Swagger(app)

    # Enable CORS globally or restrict it to certain routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Import and register blueprint
    from .routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
