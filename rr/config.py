import os

class Config:
    # Flask backend settings
    FLASK_HOST = os.getenv('FLASK_HOST', 'localhost')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/rewrite"
    
    # Gradio settings
    GRADIO_HOST = os.getenv('GRADIO_HOST', '0.0.0.0')
    GRADIO_PORT = int(os.getenv('GRADIO_PORT', 7860))
    
    # Database settings
    RATINGS_DB_PATH = os.getenv('RATINGS_DB_PATH', 'ratings.json')
    
    # LLM settings (customize based on your LLM)
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    LLM_API_KEY = os.getenv('LLM_API_KEY', 'your-api-key-here')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 1000))
