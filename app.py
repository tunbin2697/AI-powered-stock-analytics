from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from route.crawl import crawl_bp
from route.process import process_bp
from route.xai import xai_bp
from config import settings

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(crawl_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(xai_bp)
    
    # Home route
    @app.route('/')
    def index():
        """Render the main data pipeline interface"""
        return render_template('index.html')

    
    app.config.from_object(settings.Config)
    
    return app
    

if __name__ == '__main__':
    flask_app = create_app()
    host = flask_app.config.get('FLASK_RUN_HOST', '127.0.0.1')
    port = flask_app.config.get('FLASK_RUN_PORT', 5000)
    debug = flask_app.config.get('DEBUG', True)
    flask_app.run(host=host, port=port, debug=debug)