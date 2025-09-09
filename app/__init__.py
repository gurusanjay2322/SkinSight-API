from flask import Flask
from flasgger import Swagger

def create_app():
    app = Flask(__name__)

    # Swagger configuration
    app.config['SWAGGER'] = {
        'title': 'Skin Analyzer API',
        'uiversion': 3
    }

    Swagger(app)

    from .routes import bp as api_bp
    app.register_blueprint(api_bp)

    return app
