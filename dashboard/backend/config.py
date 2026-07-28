import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/models/progressive_model_xgboost_v2.pkl')
    PLAYER_DB_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/CURRENT_player_database_977_quality_FIXED.json')
    TEST_DATA_PATH = os.path.join(BASE_DIR, '../../ODI_Progressive/data/progressive_full_test_v2.csv')
    
    # API
    PORT = int(os.environ.get('PORT', '5002'))
    DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Set CORS_ORIGINS to a comma-separated list in production.
    # Defaulting to "*" keeps the public portfolio API usable after deployment.
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.environ.get('CORS_ORIGINS', '*').split(',')
        if origin.strip()
    ]

