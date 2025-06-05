from flask import Flask
from flask_cors import CORS
from config import settings
from routes.watson import watson_bp
from routes.audio import audio_bp
from routes.data_routes import data_bp
from routes.ml_routes import ml_bp
from routes.prediction_routes import prediction_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config.from_object(settings.Config)

    app.register_blueprint(watson_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(prediction_bp)

    @app.route('/')
    def health_check():
        return {"status": "Stock Chatbot Backend is running", "version": "1.0.0"}

    return app

if __name__ == '__main__':
    flask_app = create_app()
    host = flask_app.config.get('FLASK_RUN_HOST', '127.0.0.1')
    port = flask_app.config.get('FLASK_RUN_PORT', 5000)
    debug = flask_app.config.get('DEBUG', False)
    flask_app.run(host=host, port=port, debug=debug)