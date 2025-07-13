import os

class Config:
    # Set BASE_DIR to the project root (one level above 'src')
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Define SRC_DIR for paths that should be inside the 'src' folder
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Define data directories to be inside the 'src' folder
    DATA_DIR = os.path.join(SRC_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    MODELS_DIR = os.path.join(SRC_DIR, "models")

    @staticmethod
    def create_directories():
        """Creates all necessary directories defined in the Config class."""
        dirs_to_create = [
            Config.DATA_DIR,
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.MODELS_DIR
        ]
        for path in dirs_to_create:
            os.makedirs(path, exist_ok=True)
