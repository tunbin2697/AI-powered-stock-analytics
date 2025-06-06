from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from config import settings
from routes.watson import watson_bp
from routes.audio import audio_bp
from routes.data_routes import data_bp
from routes.ml_routes import ml_bp
from routes.prediction_routes import prediction_bp


def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)

    app.config.from_object(settings.Config)

    app.register_blueprint(watson_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(prediction_bp)

    # Add static file serving before the catch-all route
    @app.route('/assets/<path:filename>')
    def assets(filename):
        return send_from_directory('static/assets', filename)
    
    @app.route('/vite.svg')
    def vite_svg():
        return send_from_directory('static', 'vite.svg')

    @app.route('/')
    @app.route('/<path:path>')
    def frontend(path=''):
        # Don't serve HTML for static assets
        if path and ('.' in path.split('/')[-1]):
            # If the path looks like a file, try to serve it from static
            try:
                return send_from_directory('static', path)
            except:
                pass
        return render_template('index.html')

    return app

if __name__ == '__main__':
    flask_app = create_app()
    host = flask_app.config.get('FLASK_RUN_HOST', '127.0.0.1')
    port = flask_app.config.get('FLASK_RUN_PORT', 5000)
    debug = flask_app.config.get('DEBUG', False)
    flask_app.run(host=host, port=port, debug=debug)