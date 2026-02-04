from flask import Flask
from config import Config
from app.extensions import socketio
from pymongo import MongoClient
from flask_cors import CORS
import certifi

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Connect to MongoDB Atlas with SSL certificate
    client = MongoClient(app.config['MONGODB_URL'], tlsCAFile=certifi.where())
    app.db = client[app.config['DBNAME']]
    
    # Assets route for serving files from MongoDB GridFS
    from .routes.assets import assets_
    app.register_blueprint(assets_)
    
    from .routes.auth import auth_
    app.register_blueprint(auth_)
    
    from .routes.profile import profile_
    app.register_blueprint(profile_)
    
    from .routes.bed_selection import home_
    app.register_blueprint(home_)
    
    from .routes.similarity import similarity_
    app.register_blueprint(similarity_)
    
    from .routes.vital_prediction import prediction_
    app.register_blueprint(prediction_)
    
    from .routes.patient_treatment import treatments_
    app.register_blueprint(treatments_)
    
    from .routes.summary import summary_
    app.register_blueprint(summary_)
    
    from .routes.predict import predict_
    app.register_blueprint(predict_)
    
    socketio.init_app(app, cors_allowed_origins="*", engineio_logger=True, logger=True)

    from .websockets.patient_monitoring_socket import patientMonitoringSocketio
    app.register_blueprint(patientMonitoringSocketio)
    
    from .websockets.data_prediction_socket import dataPredictionSocketio
    app.register_blueprint(dataPredictionSocketio)
    
    from .websockets.treatment_recommendation_socket import treatmentRecommendationSocketio
    app.register_blueprint(treatmentRecommendationSocketio)
    
    # Simple health check endpoint (no model loading)
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'service': 'prism-api'}, 200
    
    # PRELOAD ML MODELS - Optimized for Railway deployment
    # Background preloading warms up worker & caches heavy libraries (torch, numpy)
    # This significantly improves page load performance with eventlet worker
    import threading
    import time
    
    def preload_ml_models():
        """
        Preload ML models in background thread with Railway-optimized strategy:
        1. Wait 10s after startup (let health check pass first)
        2. Load models with timeout protection
        3. Warm up Python VM with heavy imports (torch, numpy)
        """
        try:
            # Wait for health check to pass first (Railway timeout = 30s)
            time.sleep(10)
            print("[Startup] Background model preloading started (after health check)")
            
            from .routes.predict import get_model, get_physpol
            
            # Load models (also imports torch, numpy - warms up worker)
            get_model()      # SAC Ensemble (122MB)
            get_physpol()    # Physician policy
            
            print("[Startup] ✓ ML models preloaded successfully - worker ready!")
            
        except Exception as e:
            # Non-fatal: models will lazy-load on first request
            print(f"[Startup] Model preload failed (will lazy-load): {e}")
    
    # Start preloading in daemon thread (won't block shutdown)
    preload_thread = threading.Thread(target=preload_ml_models, daemon=True)
    preload_thread.start()
    print("[Startup] Model preloading scheduled (background thread)")
    
    return app