"""
Unified Configuration for Response Rewriter Arena
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the entire system."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Flask backend settings
    FLASK_HOST = os.getenv('FLASK_HOST', 'localhost')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/rewrite"
    FLASK_BATCH_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/batch_rewrite"
    FLASK_HEALTH_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/health"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Gradio settings
    GRADIO_HOST = os.getenv('GRADIO_HOST', '0.0.0.0')
    GRADIO_PORT = int(os.getenv('GRADIO_PORT', 7860))
    GRADIO_SHARE = os.getenv('GRADIO_SHARE', 'False').lower() == 'true'
    
    # Database settings
    RATINGS_DB_PATH = os.getenv('RATINGS_DB_PATH', str(DATA_DIR / 'ratings.json'))
    ARENA_DB_PATH = os.getenv('ARENA_DB_PATH', str(DATA_DIR / 'arena_battles.json'))
    
    # LLM settings
    LLM_PROVIDER = os.getenv('provider', 'openai')
    LLM_MODEL = os.getenv('model', 'gpt-3.5-turbo')
    LLM_API_KEY = os.getenv('LLM_API_KEY', '')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 1000))
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 30))
    
    # Cache settings
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', 1000))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour
    
    # Arena settings
    ARENA_INITIAL_ELO = int(os.getenv('ARENA_INITIAL_ELO', 1500))
    ARENA_K_FACTOR = int(os.getenv('ARENA_K_FACTOR', 32))
    ARENA_MIN_BATTLES = int(os.getenv('ARENA_MIN_BATTLES', 10))
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = str(LOGS_DIR / 'rewriter_arena.log')
    
    # API settings
    API_REQUEST_TIMEOUT = int(os.getenv('API_REQUEST_TIMEOUT', 30))
    API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', 3))
    
    # Security settings
    ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'False').lower() == 'true'
    AUTH_TOKEN = os.getenv('AUTH_TOKEN', '')
    
    # Feature flags
    ENABLE_SMART_SELECTION = os.getenv('ENABLE_SMART_SELECTION', 'True').lower() == 'true'
    ENABLE_HYBRID_PROMPT = os.getenv('ENABLE_HYBRID_PROMPT', 'True').lower() == 'true'
    ENABLE_EDGE_CASE_DETECTION = os.getenv('ENABLE_EDGE_CASE_DETECTION', 'True').lower() == 'true'
    
    # UI settings
    UI_THEME = os.getenv('UI_THEME', 'soft')
    UI_MAX_QUEUE_SIZE = int(os.getenv('UI_MAX_QUEUE_SIZE', 100))
    
    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        errors = []
        
        # Check required settings
        if not cls.LLM_PROVIDER:
            errors.append("LLM_PROVIDER is required")
        
        if not cls.LLM_MODEL:
            errors.append("LLM_MODEL is required")
        
        if cls.LLM_PROVIDER == 'openai' and not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is required for OpenAI provider")
        
        # Check paths
        if not Path(cls.BASE_DIR).exists():
            errors.append(f"Base directory does not exist: {cls.BASE_DIR}")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*50)
        print("CONFIGURATION")
        print("="*50)
        
        sections = {
            "Flask": ["FLASK_HOST", "FLASK_PORT", "DEBUG"],
            "Gradio": ["GRADIO_HOST", "GRADIO_PORT", "GRADIO_SHARE"],
            "LLM": ["LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE"],
            "Cache": ["ENABLE_CACHE", "CACHE_MAX_SIZE"],
            "Arena": ["ARENA_INITIAL_ELO", "ARENA_K_FACTOR"],
            "Features": ["ENABLE_SMART_SELECTION", "ENABLE_HYBRID_PROMPT"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = getattr(cls, key, "Not set")
                # Hide sensitive values
                if "KEY" in key or "TOKEN" in key:
                    value = "***" if value else "Not set"
                print(f"  {key}: {value}")
        
        print("\n" + "="*50)


# Create example .env file if it doesn't exist
def create_example_env():
    """Create an example .env file."""
    env_path = Path(".env.example")
    
    if not env_path.exists():
        example_content = """# Flask Backend Configuration
FLASK_HOST=localhost
FLASK_PORT=5000
DEBUG=False

# Gradio Frontend Configuration
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_SHARE=False

# LLM Configuration
provider=openai
model=gpt-3.5-turbo
LLM_API_KEY=your-api-key-here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# Database Configuration
RATINGS_DB_PATH=data/ratings.json
ARENA_DB_PATH=data/arena_battles.json

# Cache Configuration
ENABLE_CACHE=True
CACHE_MAX_SIZE=10000
CACHE_TTL=7200

# Arena Configuration
ARENA_INITIAL_ELO=1500
ARENA_K_FACTOR=32

# Feature Flags
ENABLE_SMART_SELECTION=True
ENABLE_HYBRID_PROMPT=True
ENABLE_EDGE_CASE_DETECTION=True

# Logging
LOG_LEVEL=INFO
"""
        
        with open(env_path, 'w') as f:
            f.write(example_content)
        
        print(f"Created example environment file: {env_path}")
        print("Copy to .env and update with your settings")


# Auto-create example env on import
if not Path(".env").exists() and not Path(".env.example").exists():
    create_example_env()
