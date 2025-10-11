import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/models/progressive_model_full_features.pkl')
    PLAYER_DB_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/CURRENT_player_database_977_quality.json')
    TEST_DATA_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/data/progressive_full_test.csv')
    
    # API
    PORT = 5002
    DEBUG = True
    
    # CORS - Allow localhost and ngrok
    CORS_ORIGINS = [
        'http://localhost:3000',
        'http://localhost:3001',
        'https://*.ngrok.io',
        'https://*.ngrok-free.app',
        '*'  # Allow all for ngrok (remove in production)
    ]

