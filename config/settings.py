from dotenv import load_dotenv
import os

load_dotenv()

class Config:
  DEBUG = os.getenv("DEBUG", "true")
  FLASK_RUN_HOST = os.getenv('FLASK_RUN_HOST', '127.0.0.1')
  FLASK_RUN_PORT = os.getenv('FLASK_RUN_PORT', 5000)