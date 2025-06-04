from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_default_secret_key')
    DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1']
    FLASK_RUN_HOST = os.getenv('FLASK_RUN_HOST', '127.0.0.1')
    FLASK_RUN_PORT = os.getenv('FLASK_RUN_PORT', 5000)
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API keys and other configurations
    WATSON_API_KEY = os.getenv('WATSON_API_KEY', 'your_watson_api_key')
    WATSON_URL = os.getenv('WATSON_URL', 'your_watson_url')
    
    # ML and Cache directories - all in backend project
    ML_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    SCALERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scalers')
    STOCK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stock_cache')
    
    # Create directories - ONLY PLACE that creates them
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    os.makedirs(SCALERS_DIR, exist_ok=True)
    os.makedirs(STOCK_CACHE_DIR, exist_ok=True)

    # Other settings can be added here as needed
    STOCK_API_KEY = os.getenv('STOCK_API_KEY', 'your_stock_api_key')
    STOCK_API_URL = os.getenv('STOCK_API_URL', 'https://api.example.com/stocks')